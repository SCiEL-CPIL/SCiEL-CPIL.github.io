"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_DEADLINE = 150.0  # seconds
_MARGIN = 5.0      # safety margin before deadline


def _time_left():
    return _DEADLINE - time.time()


def _energy_and_gradient(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient is with respect to the unconstrained coordinates; we project
    to the tangent space.
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # dist[i,j] = |x_i - x_j|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid log(0)
    dist = np.maximum(dist, 1e-15)
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Because d/dx_i log(1/|x_i-x_j|) = -d/dx_i log|x_i-x_j| = -(x_i-x_j)/|x_i-x_j|^2
    # So dE/dx_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    # But we need to be careful: E = sum_{i<j} log(1/d_ij)
    # dE/dx_i = sum_{j>i} d/dx_i log(1/d_ij) + sum_{j<i} d/dx_i log(1/d_ij)
    # d/dx_i log(1/d_ij) = -d/dx_i log(d_ij) = -(x_i - x_j)/d_ij^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # Wait, let me recheck.
    # log(1/d) = -log(d)
    # d/dx_i (-log(d_ij)) = - (1/d_ij) * d/dx_i (d_ij) = - (1/d_ij) * (x_i - x_j)/d_ij = -(x_i-x_j)/d_ij^2
    # So dE/dx_i = sum_{j != i} [-(x_i - x_j)/d_ij^2] = sum_{j != i} (x_j - x_i)/d_ij^2
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = dist[i, j]
            grad[i] += (pts[j] - pts[i]) / (d_ij * d_ij)
    
    # Project gradient to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad


def _energy_only(pts):
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    return -np.sum(np.log(d))


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _linalg_lbfgs(pts, max_iter=2000, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B with projection onto sphere."""
    def objective(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        return _energy_only(p)
    
    def gradient(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_gradient(p)
        return g.flatten()
    
    res = minimize(objective, pts.flatten(), jac=gradient, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
    return _project_to_sphere(res.x.reshape(-1, 3)), res.fun


def _basin_hopping(pts, n_restarts=5, step_size=0.1, n_iter=50):
    """
    Basin hopping: perturb, then polish with L-BFGS.
    """
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    for restart in range(n_restarts):
        # Perturb
        noise = np.random.randn(*pts.shape) * step_size
        perturbed = _project_to_sphere(pts + noise)
        
        # Polish
        polished, e = _linalg_lbfgs(perturbed, max_iter=1000, ftol=1e-14, gtol=1e-12)
        
        if e < best_energy:
            best_energy = e
            best_pts = polished
        
        if _time_left() < _MARGIN + 10:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    
    # Golden-angle (Fibonacci) spiral points on the unit sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # First, polish the initial configuration
    pts, e0 = _linalg_lbfgs(pts, max_iter=3000, ftol=1e-15, gtol=1e-13)
    
    if _time_left() < _MARGIN:
        return pts
    
    # Basin hopping with multiple restarts
    np.random.seed(42)
    pts, e_best = _basin_hopping(pts, n_restarts=10, step_size=0.15, n_iter=50)
    
    if _time_left() < _MARGIN:
        return pts
    
    # Final ultra-tight polish
    pts, e_final = _linalg_lbfgs(pts, max_iter=5000, ftol=1e-16, gtol=1e-14)
    
    if _time_left() < _MARGIN:
        return pts
    
    # One more basin hopping round with smaller steps for fine-tuning
    np.random.seed(123)
    pts, e_best2 = _basin_hopping(pts, n_restarts=5, step_size=0.05, n_iter=30)
    
    if _time_left() < _MARGIN:
        return pts
    
    # Final polish
    pts, e_final2 = _linalg_lbfgs(pts, max_iter=5000, ftol=1e-16, gtol=1e-14)
    
    return pts
# EVOLVE-BLOCK-END
