"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
import warnings

def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient on sphere (tangent): for each i, grad_i = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    but projected to tangent plane: grad_i -= (grad_i . x_i) x_i
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sum = np.sum(inv_dist_sq, axis=1)
    F = row_sum[:, np.newaxis] * pts - inv_dist_sq @ pts
    
    dot = np.sum(F * pts, axis=1)
    F_tangent = F - dot[:, np.newaxis] * pts
    
    return energy, F_tangent


def _energy_fast(pts):
    """Just the energy, for quick evaluation."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map_sphere(pts, tvec):
    """Riemannian exponential map on the unit sphere.
    For a tangent vector v at point x on S^2, the exponential map is:
    exp_x(v) = cos(|v|) * x + sin(|v|) / |v| * v
    This is exact for the sphere (constant curvature 1).
    """
    n = pts.shape[0]
    tv = tvec.reshape(n, 3)
    # Project to tangent space
    dots = np.sum(tv * pts, axis=1)
    tv = tv - dots[:, np.newaxis] * pts
    # Compute norm of each tangent vector
    tv_norm = np.linalg.norm(tv, axis=1, keepdims=True)
    # Handle zero vectors
    mask = tv_norm > 1e-15
    result = np.zeros_like(pts)
    # For non-zero tangent vectors: cos(|v|)*x + sin(|v|)/|v|*v
    result[mask] = (np.cos(tv_norm[mask]) * pts[mask] + 
                    np.sin(tv_norm[mask]) / tv_norm[mask] * tv[mask])
    # For zero tangent vectors: result = x
    result[~mask.ravel()] = pts[~mask.ravel()]
    return _normalize(result)


def _lbfgs_on_sphere(pts0, max_iter=2000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B on the sphere using the Riemannian exponential map."""
    n = pts0.shape[0]
    
    def _obj_and_grad(tvec):
        pts = _exp_map_sphere(pts0, tvec)
        e, g = _energy_and_grad(pts)
        return e, g.reshape(-1)
    
    res = minimize(_obj_and_grad, np.zeros(3 * n), method='L-BFGS-B',
                   jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                       'maxls': 50, 'maxfun': 5000})
    
    pts_opt = _exp_map_sphere(pts0, res.x)
    return pts_opt, res.fun, res


def _high_precision_polish(pts, time_budget=30.0):
    """Final high-precision polish with very tight tolerances."""
    start = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_fast(best_pts)
    
    # First pass: moderate precision
    try:
        pts_opt, e_opt, _ = _lbfgs_on_sphere(pts, max_iter=5000, ftol=1e-15, gtol=1e-12)
        if e_opt < best_energy:
            best_pts = pts_opt
            best_energy = e_opt
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return best_pts, best_energy
    
    # Second pass: very tight precision
    try:
        pts_opt2, e_opt2, _ = _lbfgs_on_sphere(best_pts, max_iter=10000, ftol=1e-16, gtol=1e-14)
        if e_opt2 < best_energy:
            best_pts = pts_opt2
            best_energy = e_opt2
    except Exception:
        pass
    
    return best_pts, best_energy


def _fibonacci_sphere(n):
    """Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _sobol_like_sphere(n, seed=12345):
    """Quasi-random points on sphere using a Sobol-like sequence."""
    # Use a simple low-discrepancy sequence based on golden ratio in 3D
    # Map [0,1)^3 to sphere via inverse CDF of uniform on sphere
    rng = np.random.default_rng(seed)
    # Generate quasi-random points in [0,1)^3 using a Halton-like approach
    dims = [2, 3, 5]  # coprime bases
    points = np.zeros((n, 3))
    for d in range(3):
        base = dims[d]
        for i in range(n):
            # Van der Corput sequence
            f = 1.0
            x = 1.0 / base
            val = 0.0
            idx = i + 1
            while idx > 0:
                val += (idx % base) * f
                idx //= base
                f /= base
            points[i, d] = val
    # Map to sphere: u = 2*points[:,0]-1, theta = 2*pi*points[:,1], phi = acos(points[:,2])
    # Actually use: z = 2*u-1, r = sqrt(1-z^2), theta = 2*pi*v
    z = 2.0 * points[:, 0] - 1.0
    theta = 2.0 * np.pi * points[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)


def _perturbed_fibonacci(n, rng, sigma=0.1):
    """Fibonacci points with Gaussian perturbation."""
    pts = _fibonacci_sphere(n)
    noise = rng.standard_normal((n, 3)) * sigma
    pts = pts + noise
    return _normalize(pts)


def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Pure Fibonacci
    configs.append(_fibonacci_sphere(N))
    
    # 2. Sobol-like quasi-random
    configs.append(_sobol_like_sphere(N, seed=12345))
    configs.append(_sobol_like_sphere(N, seed=67890))
    
    # 3. Perturbed Fibonacci with different noise levels
    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        configs.append(_perturbed_fibonacci(N, rng, sigma=sigma))
    
    # 4. Random configurations
    for _ in range(6):
        configs.append(_random_sphere(N, rng))
    
    # 5. Mixed configurations (Fibonacci + random)
    fib = _fibonacci_sphere(N)
    for mix in [0.3, 0.5, 0.7]:
        rand = _random_sphere(N, rng)
        configs.append(_normalize(mix * rand + (1 - mix) * fib))
    
    # Optimize each configuration
    best_pts = None
    best_energy = np.inf
    
    for i, cfg in enumerate(configs):
        if time.time() > deadline - 15.0:
            break
        
        # Allocate time based on config type
        if i == 0:
            time_budget = 12.0  # More time for Fibonacci
        elif i < 7:
            time_budget = 8.0
        else:
            time_budget = 6.0
        
        pts_opt, e_opt = _high_precision_polish(cfg, time_budget=time_budget)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Final polish with remaining time
    if best_pts is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 8.0:
            best_pts, best_energy = _high_precision_polish(best_pts, time_budget=remaining)
    
    # Ensure points are well-separated and on the sphere
    best_pts = _normalize(best_pts)
    
    # Validate: check no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure all finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
