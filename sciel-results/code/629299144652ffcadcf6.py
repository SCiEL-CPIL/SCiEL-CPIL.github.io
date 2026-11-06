"""Optimized 100-point log-energy minimization on S^2 using multi-start Nesterov Momentum."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Compute squared distances
    # d2[i,j] = |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2 x_i . x_j
    # Since points are on unit sphere, |x_i|^2 = 1
    # d2[i,j] = 2 - 2 * (x_i . x_j)
    dots = np.dot(pts, pts.T)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0) # Avoid log(0)
    
    # Extract upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d2_upper = d2[mask]
    
    # Check for very small distances
    if np.any(d2_upper < 1e-15):
        return -1e18
        
    # Energy is sum -log(d) = -0.5 * sum log(d2)
    # Using math.fsum for precision as per contract
    log_vals = np.log(d2_upper)
    # The contract says sum log(1/d) = - sum log(d)
    # log(1/d) = -0.5 log(d2)
    # So energy = -0.5 * fsum(log(d2_upper))
    return -0.5 * math.fsum(log_vals.tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere."""
    n = len(pts)
    dots = np.dot(pts, pts.T)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d2_upper = d2[mask]
    
    if np.any(d2_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    
    # Energy
    log_vals = np.log(d2_upper)
    energy = -0.5 * math.fsum(log_vals.tolist())
    
    # Gradient
    # E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    # d/dx_i log(|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    # So dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Note: The term for j=i is 0, so we can sum over all j.
    
    # inv_d2[i,j] = 1 / |x_i - x_j|^2
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # grad[i] = - sum_j inv_d2[i,j] * diff[i,j]
    # This is a matrix multiplication: - (inv_d2 @ diff)
    # But diff is 3D. We need to sum over j for each i.
    # grad[i,k] = - sum_j inv_d2[i,j] * diff[i,j,k]
    # We can do this with einsum: grad[i,k] = - sum_j inv_d2[i,j] * (pts[i,k] - pts[j,k])
    # grad[i,k] = - sum_j inv_d2[i,j]*pts[i,k] + sum_j inv_d2[i,j]*pts[j,k]
    # sum_j inv_d2[i,j]*pts[i,k] = pts[i,k] * sum_j inv_d2[i,j]
    # sum_j inv_d2[i,j]*pts[j,k] = (inv_d2 @ pts)[i,k]
    
    row_sums = np.sum(inv_d2, axis=1)
    term1 = pts * row_sums[:, np.newaxis]
    term2 = inv_d2 @ pts
    
    grad = -(term1 - term2)
    
    # Project gradient onto tangent space
    # g_tan = g - (g . x) x
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points based on icosahedron symmetry."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    pts = []
    for i in range(n):
        v = verts[i % len(verts)]
        # Add small noise to break symmetry
        noise = np.random.RandomState(i + 1000).randn(3) * 0.05
        v = v + noise
        v /= np.linalg.norm(v)
        pts.append(v)
    return np.array(pts)

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _optimize_nesterov(pts0, deadline, max_iter=20000, initial_lr=0.01):
    """Optimize using Nesterov Momentum with adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = initial_lr
    momentum = 0.9
    
    # Line search parameters
    c1 = 1e-4
    rho = 0.5
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
            
        # Nesterov lookahead
        pts_lookahead = pts + momentum * v
        _, grad_la = _log_energy_grad(pts_lookahead)
        
        # Adaptive learning rate
        # If gradient norm is large, reduce lr; if small, increase slightly
        # But primarily rely on line search
        
        # Backtracking line search
        t = lr
        while t > 1e-12:
            new_pts = pts - t * grad_la
            new_pts = _project_to_sphere(new_pts)
            
            # Check if energy decreased sufficiently (Armijo condition)
            # We approximate the directional derivative
            # For simplicity, just check if energy is lower
            e_new = _log_energy(new_pts)
            
            if e_new < e - c1 * t * np.dot(grad_la, new_pts - pts):
                break
            t *= rho
            
        # Update velocity and position
        v = momentum * v - t * grad_la
        pts = pts + v
        pts = _project_to_sphere(pts)
        
        # Adaptive momentum: increase as we converge
        # momentum = min(0.99, 0.9 + 0.09 * (it / max_iter))
        
        # Occasionally reset velocity if stuck
        if it % 500 == 0 and it > 0:
             # Check if we are stuck
             pass # For now, keep it simple

    return best_pts, best_energy

def _optimize_lbfgs(pts0, deadline, maxiter=1000):
    """Optimize using L-BFGS-B."""
    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _log_energy_grad(pts)
        return g.flatten()

    try:
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'maxfun': maxiter * 4,
                                   'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _log_energy(best_pts)
            return best_pts, best_energy
    except Exception:
        pass
        
    return pts0, _log_energy(pts0)

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    # Safety margin: 150s limit, leave 5s for overhead/return
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    # Generate diverse starting configurations
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_icosahedron_sphere(N))
    
    # Add random starts
    for seed in [42, 123, 999, 777, 555, 314, 271, 161]:
        starts.append(_random_sphere(N, seed=seed))
        
    # Phase 1: Quick screening with L-BFGS-B
    # Allocate ~30s for this phase
    phase1_deadline = start_time + 30.0
    for pts0 in starts:
        if time.time() > phase1_deadline:
            break
        pts_opt, e_opt = _optimize_lbfgs(pts0, phase1_deadline, maxiter=200)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
            
    # Phase 2: Deep refinement with Nesterov Momentum
    # Allocate remaining time
    if best_pts is not None:
        # Run Nesterov on the best candidate
        pts_nest, e_nest = _optimize_nesterov(best_pts, deadline, max_iter=50000, initial_lr=0.01)
        if e_nest < best_energy:
            best_energy = e_nest
            best_pts = pts_nest
            
        # If time allows, try Nesterov on a few other good candidates from Phase 1
        # We didn't store all Phase 1 results, so we just stick to the best one.
        # Alternatively, we could run Nesterov on the 2nd best, but we need to track them.
        
    # Final polish with L-BFGS-B if time remains
    if best_pts is not None and time.time() < deadline - 10:
        pts_final, e_final = _optimize_lbfgs(best_pts, deadline, maxiter=500)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        
    best_pts = _project_to_sphere(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
