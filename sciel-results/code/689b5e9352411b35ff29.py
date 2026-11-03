"""Minimal logarithmic energy of 100 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax

N = 100
DEADLINE = 115.0  # safety margin before 120s kill
START_TIME = time.time()


def _remaining():
    return DEADLINE - (time.time() - START_TIME)


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * (-1)
                         = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    But since we minimize sum log(1/d) = -sum log(d), 
    d/dx_i [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    So grad_i = sum_{j!=i} (xj - xi)/|xi-xj|^2 ... wait let me re-derive.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = -sum_{j != i} d/dxi log|xi-xj| = -sum_{j != i} (xi-xj)/|xi-xj|^2
    """
    # Compute pairwise distances
    # pts shape (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/d) = -0.5 * sum_{i!=j} log(d_ij)
    # Using fsum for precision
    # We'll compute the full sum and halve
    # log(1/d) = -log(d)
    log_dist = np.log(dist)
    # Set diagonal to 0
    np.fill_diagonal(log_dist, 0.0)
    # Sum over upper triangle only for the pair sum
    # Actually, let's just do sum_{i<j} -log(d_ij)
    # Use the full matrix sum / 2
    total_log = np.sum(log_dist)  # this is sum over all i!=j of log(d_ij)
    energy = -0.5 * total_log
    
    # Gradient: grad_i = -sum_{j != i} (xi - xj) / |xi - xj|^2
    # = sum_{j != i} (xj - xi) / dist_ij^2
    # diff[i,j] = xi - xj, so we want -(diff[i,j]) / dist_sq[i,j]
    # grad[i] = -sum_j diff[i,j] / dist_sq[i,j] (j != i)
    # Since dist_sq diagonal is set to 1 and diff diagonal is 0, no issue
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _energy_only(pts):
    """Compute just the energy for scipy minimize."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    return -0.5 * np.sum(log_dist)


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_init(seed=0):
    """Golden-angle spiral initialization."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_init(seed=0):
    """Random initialization on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _sobol_init(seed=0):
    """Sobol sequence initialization for better space coverage."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Need at least 2^k points; use 128 and take first 100
    pts = sampler.random(N)
    # Map from [0,1]^3 to sphere
    # Use inverse CDF for uniform on sphere
    z = 2.0 * pts[:, 2] - 1.0
    phi = 2.0 * math.pi * pts[:, 0]
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return result


def _optimize_from_start(pts0, maxiter=200, gtol=1e-8):
    """Run BFGS optimization from a starting point, projecting to sphere each step."""
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Project to sphere
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . x) * x
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_proj.flatten()
    
    x0 = pts0.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='BFGS',
        options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
    )
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    best_energy = _energy_only(best_pts)
    return best_pts, best_energy


def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    inits = []
    
    # Fibonacci spiral (good baseline)
    inits.append(_fibonacci_init(seed=0))
    
    # Sobol-based initializations with different seeds
    for seed in range(10):
        if _remaining() < 30.0:
            break
        inits.append(_sobol_init(seed=seed))
    
    # Random initializations
    for seed in range(5):
        if _remaining() < 30.0:
            break
        inits.append(_random_init(seed=100 + seed))
    
    # Optimize from each starting point
    for idx, pts0 in enumerate(inits):
        if _remaining() < 15.0:
            break
        
        try:
            pts_opt, energy_opt = _optimize_from_start(pts0, maxiter=300, gtol=1e-9)
            if energy_opt < best_energy:
                best_energy = energy_opt
                best_pts = pts_opt
        except Exception:
            continue
        
        # If we have time, do a second round with tighter tolerance
        if _remaining() > 60.0 and idx < 5:
            try:
                pts_opt2, energy_opt2 = _optimize_from_start(pts_opt, maxiter=500, gtol=1e-10)
                if energy_opt2 < best_energy:
                    best_energy = energy_opt2
                    best_pts = pts_opt2
            except Exception:
                pass
    
    # Final polish on the best result
    if best_pts is not None and _remaining() > 20.0:
        try:
            pts_final, energy_final = _optimize_from_start(best_pts, maxiter=1000, gtol=1e-11)
            if energy_final < best_energy:
                best_pts = pts_final
                best_energy = energy_final
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_init(seed=0)
    
    # Ensure all points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_init(seed=0)
        best_pts = _project_to_sphere(best_pts)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_init(seed=0)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
