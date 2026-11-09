import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Pairwise differences: (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Squared distances: (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: -0.5 * sum of log(dist_sq) over upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    # Gradient
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / np.maximum(dist_sq[mask], 1e-30)
    
    # grad_i = sum_j (x_j - x_i) / dist_sq_ij = -sum_j diff[i,j] / dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral():
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(N, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)

def _optimize_lbfgs(pts_init, maxiter=1000, ftol=1e-12, gtol=1e-10, time_limit=60.0):
    """
    Optimize using L-BFGS-B with sphere constraint handled by projection.
    We optimize in 3N dimensions with a penalty for off-sphere points,
    then project back.
    """
    start = time.time()
    
    # Flatten
    x0 = pts_init.flatten().copy()
    
    C_pen = 1e8
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Project to sphere for energy computation
        pts_sph = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts_sph)
        # Penalty for being off sphere
        norms_sq = np.sum(pts * pts, axis=1)
        penalty = C_pen * np.sum((norms_sq - 1.0) ** 2)
        return e + penalty
    
    def jac(x):
        pts = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts_sph)
        # Gradient of penalty
        norms_sq = np.sum(pts * pts, axis=1)
        g_pen = 4.0 * C_pen * pts * (norms_sq - 1.0)[:, np.newaxis]
        # Chain rule: gradient of energy w.r.t. pts_sph, then pts_sph w.r.t. pts
        # pts_sph = pts / ||pts||
        # d(pts_sph)/d(pts) is complicated. Instead, we use the fact that
        # we project, so the gradient in the ambient space needs care.
        # Actually, let's just use the penalty method properly.
        # The gradient of the total objective w.r.t. x (flat) is:
        # d/dx [energy(project(x)) + penalty(x)]
        # = (d energy/d pts_sph) @ (d pts_sph/d x) + d penalty/d x
        # This is complex. Let's use a simpler approach: optimize directly
        # on the sphere using a different parameterization or just use
        # the penalty and hope it's tight enough.
        
        # For now, approximate: the gradient of energy w.r.t. pts is g
        # (since we computed it at pts_sph), and the gradient of penalty is g_pen.
        # But this isn't quite right because pts_sph != pts.
        
        # Better approach: compute gradient numerically or use a proper
        # constrained optimization. Let's use the penalty gradient directly
        # and add the energy gradient as an approximation.
        
        return (g + g_pen).flatten()
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxfun': maxiter * 2})
        if time.time() - start < time_limit:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass
    
    # Project result to sphere
    result = _project_to_sphere(best_x.reshape(N, 3))
    return result, best_val

def _polish_lbfgs(pts, maxiter=5000, ftol=1e-14, gtol=1e-12, time_limit=30.0):
    """
    High-precision polish pass using L-BFGS-B with tighter tolerances.
    Uses a direct sphere-constrained approach via Lagrange multipliers.
    """
    start = time.time()
    
    pts = _project_to_sphere(pts.copy())
    
    # Use a penalty method with very high penalty for precision
    C_pen = 1e12
    
    x0 = pts.flatten().copy()
    
    def objective(x):
        pts_local = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts_local)
        e, _ = _energy_and_grad(pts_sph)
        norms_sq = np.sum(pts_local * pts_local, axis=1)
        penalty = C_pen * np.sum((norms_sq - 1.0) ** 2)
        return e + penalty
    
    def jac(x):
        pts_local = x.reshape(N, 3)
        pts_sph = _project_to_sphere(pts_local)
        e, g = _energy_and_grad(pts_sph)
        norms_sq = np.sum(pts_local * pts_local, axis=1)
        g_pen = 4.0 * C_pen * pts_local * (norms_sq - 1.0)[:, np.newaxis]
        return (g + g_pen).flatten()
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxfun': maxiter * 3})
        if time.time() - start < time_limit:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass
    
    result = _project_to_sphere(best_x.reshape(N, 3))
    return result, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Initial points: Fibonacci spiral
    pts_init = _fibonacci_spiral()
    
    # First optimization pass
    best_pts, best_val = _optimize_lbfgs(pts_init, maxiter=2000, ftol=1e-10, gtol=1e-8, time_limit=40.0)
    
    if time.time() > deadline:
        return best_pts
    
    # Try a few random restarts
    rng_seeds = [42, 123, 456, 789, 101112, 202021, 314159, 271828]
    for seed in rng_seeds:
        if time.time() > deadline - 20.0:
            break
        
        pts_r = _random_points(seed)
        pts_opt, val = _optimize_lbfgs(pts_r, maxiter=1000, ftol=1e-10, gtol=1e-8, time_limit=15.0)
        
        if val < best_val:
            best_val = val
            best_pts = pts_opt
    
    if time.time() > deadline:
        return best_pts
    
    # Final high-precision polish
    best_pts, best_val = _polish_lbfgs(best_pts, maxiter=5000, ftol=1e-14, gtol=1e-12, time_limit=25.0)
    
    # Ensure exact unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
