"""
Strategy: Direct Cartesian Tangent-Space Optimization.
- Parameterize points directly as Cartesian vectors on the sphere.
- Use a custom L-BFGS-B implementation that projects gradients to the tangent space.
- This avoids the singularities of spherical coordinates at the poles and the
  distortion of the metric.
- Multi-restart with "cooling" perturbations: start with large random noise,
  then progressively smaller noise around the best solution found so far,
  effectively performing a stochastic local search.
- Exploit the 120s budget for many iterations.
"""

import numpy as np
from scipy.optimize import minimize
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.92  # Safety margin for 120s limit (approx 110s)

def _energy_and_grad_cartesian(pts):
    """
    Compute logarithmic energy and gradient in Cartesian coordinates.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    d_ij = ||x_i - x_j||
    
    Gradient in R^3 (before projection):
    grad_x_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    
    Note: Since points are constrained to the sphere, the true gradient 
    on the manifold is the projection of this Euclidean gradient onto 
    the tangent space at x_i.
    proj(v) = v - (v . x_i) x_i
    """
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    mask = dists > 1e-12
    inv_d = np.zeros_like(dists)
    inv_d[mask] = 1.0 / dists[mask]
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient accumulation
    # grad_cart[i] = sum_{j!=i} (x_i - x_j) / d_ij^2
    # coeff[i,j] = 1/d_ij^2 if i!=j else 0
    coeff = np.zeros_like(dists_sq)
    mask_sq = dists_sq > 1e-24
    coeff[mask_sq] = 1.0 / dists_sq[mask_sq]
    np.fill_diagonal(coeff, 0.0)
    
    # grad_cart[i] = sum_j coeff[i,j] * (x_i - x_j)
    # = x_i * sum_j coeff[i,j] - sum_j coeff[i,j] * x_j
    sum_coeff = np.sum(coeff, axis=1)
    sum_coeff_pts = coeff @ pts
    
    grad_cart = pts * sum_coeff[:, np.newaxis] - sum_coeff_pts
    
    # Project gradient to tangent space: g_tan = g - (g . x) x
    # Since x is unit vector, x . x = 1
    dot_gx = np.sum(grad_cart * pts, axis=1)
    grad_tangent = grad_cart - dot_gx[:, np.newaxis] * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points in Cartesian coordinates."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])

def _random_points(n, rng):
    """Generate random points on the sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def construct():
    start_time = time.time()
    total_time = 120.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = 1e30
    
    # Initial guess: Fibonacci spiral
    fib_pts = _fibonacci_spiral(N)
    best_pts = fib_pts.copy()
    best_energy, _ = _energy_and_grad_cartesian(best_pts)
    
    # Define objective and gradient for scipy
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad_cartesian(pts)
        return e
        
    def grad_func(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        _, g = _energy_and_grad_cartesian(pts)
        return g.flatten()
        
    # Multi-restart strategy
    # We will try a series of restarts.
    # Phase 1: Broad exploration with random starts and perturbed Fibonacci.
    # Phase 2: Focused exploitation around the best solution found.
    
    n_restarts = 0
    # We estimate ~0.5-1s per L-BFGS run. Let's allow up to 60 restarts.
    # But we will stop when time runs out.
    
    # Strategy:
    # 1. Run a few random restarts.
    # 2. Run perturbed Fibonacci restarts with decreasing noise.
    # 3. Polish the best.
    
    restart_configs = []
    # 5 random restarts
    for i in range(5):
        restart_configs.append(('random', 0.0))
    # 20 perturbed Fibonacci restarts with decreasing noise
    # Noise scales: 0.5, 0.4, ..., 0.1, then smaller
    noise_scales = np.linspace(0.5, 0.05, 20)
    for scale in noise_scales:
        restart_configs.append(('fib_perturb', scale))
    
    # Shuffle random restarts to mix them in? No, keep structured.
    # Actually, let's interleave them or just do them in order.
    # Order: Random first to find good basins, then refine.
    
    for config in restart_configs:
        if time.time() > deadline - 5:  # Leave 5s for final polish
            break
            
        n_restarts += 1
        kind, scale = config
        
        if kind == 'random':
            init_pts = _random_points(N, rng)
        elif kind == 'fib_perturb':
            init_pts = fib_pts + rng.standard_normal((N, 3)) * scale
            init_pts = _project_to_sphere(init_pts)
        
        try:
            # Use L-BFGS-B. 
            # We don't have explicit bounds, but the projection handles the constraint.
            # However, L-BFGS-B might wander far from the sphere.
            # The objective function projects internally, so it's smooth enough?
            # Actually, the projection makes the objective non-smooth at the origin,
            # but we are far from it.
            # A better approach: Use BFGS on the sphere? No, scipy doesn't do manifolds.
            # Using L-BFGS-B on the embedded space with projection in the objective
            # is standard but can be slow if it wanders.
            # Let's stick to L-BFGS-B with the projected gradient.
            
            res = minimize(
                objective, init_pts.flatten(), jac=grad_func, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4, 'maxls': 20}
            )
            cand_pts = _project_to_sphere(res.x.reshape(N, 3))
            
            cand_energy, _ = _energy_and_grad_cartesian(cand_pts)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_pts.copy()
                
        except Exception:
            continue
            
    # Final Polish
    if time.time() < deadline - 1 and best_pts is not None:
        try:
            # Tighter tolerances, more iterations
            res = minimize(
                objective, best_pts.flatten(), jac=grad_func, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            cand_pts = _project_to_sphere(res.x.reshape(N, 3))
            cand_energy, _ = _energy_and_grad_cartesian(cand_pts)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_pts.copy()
        except Exception:
            pass
            
    # Return best points
    # Ensure they are strictly on the sphere
    return _project_to_sphere(best_pts)
