"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 110.0  # safety margin before 120s kill


def _energy_and_grad(pts):
    """Compute log-energy and its gradient for points on the unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  for each i
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Energy: sum over i<j of -log(dist_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / dist_ij^2
    inv_dist2 = np.zeros_like(dist)
    inv_dist2[iu] = 1.0 / (dist[iu] ** 2)
    inv_dist2 = (inv_dist2 + inv_dist2.T)  # symmetric
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j inv_dist2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_dist2[i,j] * pts[j] - pts[i] * sum_j inv_dist2[i,j]
    grad = (inv_dist2 @ pts) - pts * inv_dist2.sum(axis=1, keepdims=True)
    
    return energy, grad


def _fibonacci_sphere(n, seed=0):
    """Generate n points on unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed * 0.1
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_and_normalize(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_from_start(pts0):
    """Run L-BFGS-B from an initial configuration, returning best energy and points."""
    pts = pts0.copy()
    
    def objective(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        # Project gradient to tangent space: g_tan = g - (g·p_hat)p_hat
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        return g_tan.ravel()
    
    x0 = pts0.ravel()
    result = minimize(
        objective, x0, jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    pts_opt = _project_and_normalize(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(pts_opt)
    return energy, pts_opt


def construct():
    """Multi-start optimization of logarithmic energy on the unit sphere."""
    t_start = time.time()
    best_energy = float('inf')
    best_pts = None
    
    # Start 1: Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    # Check time
    if time.time() > _DEADLINE - 30:
        return best_pts
    
    # Start 2: Random with seed 42
    rng = np.random.default_rng(42)
    pts0 = _random_sphere(N, rng)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    if time.time() > _DEADLINE - 20:
        return best_pts
    
    # Start 3: Fibonacci with offset
    pts0 = _fibonacci_sphere(N, seed=1.0)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    if time.time() > _DEADLINE - 10:
        return best_pts
    
    # Start 4: Another random
    rng = np.random.default_rng(123)
    pts0 = _random_sphere(N, rng)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    return best_pts
# EVOLVE-BLOCK-END
