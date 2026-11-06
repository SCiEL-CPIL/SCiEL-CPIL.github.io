"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid log(0) by masking diagonal
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    log_terms = np.log(1.0 / dists)
    return math.fsum(log_terms[mask].tolist())


def _log_energy_grad(pts):
    """Compute gradient of log-energy w.r.t. point positions (in R^3) vectorized.
    d/dx_i log(1/|xi-xj|) = -(xi-xj)/|xi-xj|^2 for each j != i
    Sum over j != i.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    # grad[i] = -sum_{j != i} (pts[i] - pts[j]) / |pts[i]-pts[j]|^2
    # = -sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    inv_dist_sq[np.diag_indices(n)] = 0.0
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed=0):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_lbfgs(pts, time_limit=30.0):
    """Optimize using L-BFGS-B on the sphere via parameterization."""
    start_time = time.time()
    
    def objective_and_grad(flat):
        pts = flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e = _log_energy(pts)
        g = _log_energy_grad(pts)
        # Project gradient to tangent space: g_tangent = g - (g·x)x
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_proj.flatten()
    
    x0 = pts.flatten()
    e0, g0 = objective_and_grad(x0)
    
    try:
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 50}
        )
        pts_opt = _project_to_sphere(result.x.reshape(N, 3))
    except Exception:
        pts_opt = pts
    
    return pts_opt


def _optimize_gradient_descent(pts, lr=0.001, max_iter=5000, time_limit=30.0):
    """Optimize using gradient descent with momentum on the sphere."""
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    velocity = np.zeros_like(pts)
    momentum = 0.9
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        g = _log_energy_grad(pts)
        # Project to tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        velocity = momentum * velocity - lr * g_tangent
        pts_new = pts + velocity
        pts_new = _project_to_sphere(pts_new)
        
        e = _log_energy(pts_new)
        if e < best_energy:
            best_energy = e
            best_pts = pts_new.copy()
        pts = pts_new
    
    return best_pts


def _steepest_descent(pts, time_limit=20.0, tol=1e-14):
    """Steepest descent with backtracking line search on the sphere."""
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        g = _log_energy_grad(pts)
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        grad_norm = np.linalg.norm(g_tangent)
        
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step = 0.01
        g_tangent_norm = np.linalg.norm(g_tangent)
        if g_tangent_norm > 0:
            step = 1.0 / (g_tangent_norm * g_tangent_norm)
            step = min(step, 0.1)
        
        for _ in range(30):
            pts_new = pts - step * g_tangent
            pts_new = _project_to_sphere(pts_new)
            e_new = _log_energy(pts_new)
            if e_new < best_energy - 1e-15 * step * grad_norm:
                pts = pts_new
                best_energy = e_new
                best_pts = pts_new.copy()
                break
            step *= 0.5
        
        if step < 1e-15:
            break
    
    return best_pts


def _polish_high_precision(pts, time_limit=20.0):
    """Final high-precision polish using L-BFGS with tight tolerances."""
    start_time = time.time()
    
    def objective_and_grad(flat):
        pts = flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e = _log_energy(pts)
        g = _log_energy_grad(pts)
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_proj.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-18, 'maxls': 100}
        )
        pts_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        e_init = _log_energy(pts)
        if e_opt < e_init:
            return pts_opt
        else:
            return pts
    except Exception:
        return pts


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Phase 1: Try multiple initial configurations with L-BFGS
    initializations = [
        _fibonacci_sphere(),
        _random_sphere(seed=42),
        _random_sphere(seed=123),
        _random_sphere(seed=777),
        _random_sphere(seed=2024),
        _random_sphere(seed=31415),
    ]
    
    best_pts = None
    best_energy = float('inf')
    
    for init in initializations:
        if time.time() > deadline - 60.0:
            break
        
        # L-BFGS optimization
        try:
            opt_pts = _optimize_lbfgs(init, time_limit=20.0)
            e = _log_energy(opt_pts)
            if e < best_energy:
                best_energy = e
                best_pts = opt_pts
        except Exception:
            continue
    
    if best_pts is None:
        return _fibonacci_sphere()
    
    # Phase 2: Steepest descent with backtracking
    if time.time() < deadline - 40.0:
        sd_pts = _steepest_descent(best_pts, time_limit=25.0)
        e_sd = _log_energy(sd_pts)
        if e_sd < best_energy:
            best_energy = e_sd
            best_pts = sd_pts
    
    # Phase 3: Gradient descent with momentum
    if time.time() < deadline - 25.0:
        gd_pts = _optimize_gradient_descent(best_pts, lr=0.0005, max_iter=2000, time_limit=15.0)
        e_gd = _log_energy(gd_pts)
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = gd_pts
    
    # Phase 4: High-precision L-BFGS polish
    if time.time() < deadline - 10.0:
        polished = _polish_high_precision(best_pts, time_limit=10.0)
        e_pol = _log_energy(polished)
        if e_pol < best_energy:
            best_energy = e_pol
            best_pts = polished
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    # Check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fallback to Fibonacci if points are too close
        return _fibonacci_sphere()
    
    return best_pts
# EVOLVE-BLOCK-END
