"""Optimized logarithmic energy minimization for 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0  # wall-clock limit from evaluator
SAFETY_MARGIN = 5.0  # seconds to leave before deadline

def _log_energy(pts):
    """Compute log(1/d) energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log of very small numbers causing issues
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _log_energy_and_grad(pts):
    """Compute log(1/d) energy and its gradient with respect to points on sphere."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each pair (i,j) with i<j:
    #   grad_i += (x_i - x_j) / d_ij^2
    #   grad_j += (x_j - x_i) / d_ij^2
    grad = np.zeros_like(pts)
    ii, jj = iu
    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        dij = d[k]
        vec = pts[i] - pts[j]
        grad[i] += vec / (dij * dij)
        grad[j] += -vec / (dij * dij)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Try L-BFGS-B optimization with multiple strategies
    def objective_and_grad(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _log_energy_and_grad(p)
        # Project gradient to tangent space: g - (g·p)p
        gp = np.sum(g * p, axis=1, keepdims=True)
        g_tangent = g - gp * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    
    # First optimization pass
    if time.time() < deadline:
        try:
            res = minimize(
                objective_and_grad, x0, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
            )
            pts_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Second pass: refine with tighter tolerance
    if time.time() < deadline:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective_and_grad, x0, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 100}
            )
            pts_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Try from random starts if time permits
    for seed in range(10):
        if time.time() >= deadline:
            break
        x0 = _random_sphere(N, seed * 42 + 7).flatten()
        try:
            res = minimize(
                objective_and_grad, x0, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
            )
            pts_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Final polish if time remains
    if time.time() < deadline:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective_and_grad, x0, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-18, 'maxls': 200}
            )
            pts_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
