"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only sum upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    energy = np.sum(np.log(1.0 / dists[iu]))
    return energy

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. each point.
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-30:
                continue
            grad[i] += d / dist_sq
    return grad

def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Use L-BFGS-B with sphere projection
    from scipy.optimize import minimize
    
    def objective_and_grad(flat_pts):
        pts_3d = flat_pts.reshape(N, 3)
        # Project to sphere
        pts_3d = _project_to_sphere(pts_3d)
        e = _log_energy(pts_3d)
        g = _log_energy_grad(pts_3d)
        # Project gradient to tangent space: g - (g . x) x
        g_tangent = g - np.sum(g * pts_3d, axis=1, keepdims=True) * pts_3d
        return e, g_tangent.ravel()
    
    # Try L-BFGS-B
    try:
        def fun_and_grad(x):
            e, g = objective_and_grad(x)
            return e, g
        
        res = minimize(fun_and_grad, pts.ravel(), method='L-BFGS-B', 
                      jac=True, options={'maxiter': 2000, 'maxfun': 20000,
                                        'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxls': 50})
        
        if res.success or res.fun < best_energy:
            best_pts = _project_to_sphere(res.x.reshape(N, 3))
            best_energy = res.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return best_pts
    
    # If we have time, try a few more restarts or local refinements
    # Use simple gradient descent with projection as fallback
    if time.time() < deadline - 10:
        # Try a few random perturbations and local optimization
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline:
                break
            # Perturb current best
            noise = rng.normal(0, 0.1, size=(N, 3))
            pts_try = _project_to_sphere(best_pts + noise)
            e_try = _log_energy(pts_try)
            
            if e_try < best_energy:
                best_pts = pts_try
                best_energy = e_try
            
            # Local gradient descent
            step_size = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                g = _log_energy_grad(pts_try)
                # Project gradient to tangent space
                g_tangent = g - np.sum(g * pts_try, axis=1, keepdims=True) * pts_try
                pts_try = _project_to_sphere(pts_try - step_size * g_tangent)
                e_try = _log_energy(pts_try)
                if e_try < best_energy:
                    best_pts = pts_try.copy()
                    best_energy = e_try
                # Adaptive step size
                if step % 50 == 0 and step > 0:
                    # Check if step is too large
                    if not np.all(np.isfinite(pts_try)):
                        step_size *= 0.5
                        break
    
    return best_pts
# EVOLVE-BLOCK-END
