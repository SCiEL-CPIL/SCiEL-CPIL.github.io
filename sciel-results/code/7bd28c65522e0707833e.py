"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total

def _energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only use upper triangle
    iu = np.triu_indices(n, 1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _energy_and_grad(pts):
    """Compute energy and gradient (in R^3, then projected)."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i+1, n):
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                return 1e10, np.zeros_like(pts)
            # dE/dd = -1/d, grad_d E = -d_vec / d^2
            term = -1.0 / d
            grad[i] += term * d_vec
            grad[j] -= term * d_vec
            energy += math.log(1.0 / d)
    # Project gradient to tangent space: grad_tan = grad - (grad . x) * x
    grad_proj = grad.copy()
    for i in range(n):
        dot = np.dot(grad[i], pts[i])
        grad_proj[i] = grad[i] - dot * pts[i]
    return energy, grad_proj

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(rng):
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def _optimize_from_start(pts, max_time=100.0):
    """Optimize from a given starting configuration."""
    start_time = time.time()
    current_energy = _energy_fast(pts)
    current_pts = pts.copy()
    
    # Use L-BFGS-B in R^3 with projection, or use a custom tangent-space approach
    # Simpler: use scipy minimize with constraints handled by projection
    best_energy = current_energy
    best_pts = current_pts.copy()
    
    # Multiple L-BFGS-B restarts with decreasing maxiter
    remaining = max_time - (time.time() - start_time)
    if remaining < 5:
        return best_pts, best_energy
    
    # First optimization with more iterations
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # We need gradient w.r.t. x (before projection)
        # Approximate: use the projected gradient, which is a reasonable direction
        # For L-BFGS-B, we'll just use the raw gradient and let projection handle constraints
        return e, g.reshape(-1)
    
    try:
        res = minimize(
            objective_and_grad,
            current_pts.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'gtol': 1e-6, 'maxfun': 500}
        )
        new_pts = _project_to_sphere(res.x.reshape(N, 3))
        new_energy = _energy_fast(new_pts)
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts
    except Exception:
        pass
    
    # Check time
    remaining = max_time - (time.time() - start_time)
    if remaining < 5:
        return best_pts, best_energy
    
    # Second optimization from the improved point with tighter tolerance
    try:
        res2 = minimize(
            objective_and_grad,
            best_pts.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-8, 'maxfun': 1000}
        )
        new_pts2 = _project_to_sphere(res2.x.reshape(N, 3))
        new_energy2 = _energy_fast(new_pts2)
        if new_energy2 < best_energy:
            best_energy = new_energy2
            best_pts = new_pts2
    except Exception:
        pass
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_spiral()
    
    best_pts = pts0.copy()
    best_energy = _energy_fast(pts0)
    
    # Optimize from Fibonacci start
    remaining = deadline - time.time()
    if remaining > 10:
        opt_pts, opt_energy = _optimize_from_start(pts0, max_time=min(remaining - 2, 60))
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    n_restarts = 3
    for k in range(n_restarts):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        pts_r = _random_points(rng)
        # Quick local optimization with limited budget
        time_budget = min(remaining - 3, 20)
        if time_budget < 5:
            break
        opt_pts_r, opt_energy_r = _optimize_from_start(pts_r, max_time=time_budget)
        if opt_energy_r < best_energy:
            best_energy = opt_energy_r
            best_pts = opt_pts_r
    
    # Final check
    remaining = deadline - time.time()
    if remaining > 5:
        # One more polish
        opt_pts_f, opt_energy_f = _optimize_from_start(best_pts, max_time=min(remaining - 2, 15))
        if opt_energy_f < best_energy:
            best_energy = opt_energy_f
            best_pts = opt_pts_f
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-6 * rng.standard_normal(3)
                best_pts = _project_to_sphere(best_pts)
                break
    
    return best_pts
# EVOLVE-BLOCK-END
