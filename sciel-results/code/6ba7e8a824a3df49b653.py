"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _energy_and_grad_cartesian(pts):
    """Compute log energy and projected gradient in Cartesian coordinates."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / dist_ij^2
    # We want the projected gradient on the sphere: P_i(g_i) = g_i - (g_i . x_i) x_i
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_j (x_i - x_j) / dist_ij^2 = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: P(g) = g - (g . x) x
    dot = np.sum(grad * pts, axis=1)
    grad_proj = grad - dot[:, np.newaxis] * pts
    
    return energy, grad_proj

def _lbfgs_sphere(pts, maxiter=5000, ftol=1e-14, gtol=1e-12, time_limit=None):
    """L-BFGS-B optimization on the sphere using Cartesian coordinates."""
    start_time = time.time()
    deadline = start_time + time_limit if time_limit else float('inf')
    
    def objective_and_grad(x_flat):
        if time.time() > deadline:
            raise TimeoutError()
        pts = x_flat.reshape(-1, 3)
        e, g = _energy_and_grad_cartesian(pts)
        return e, g
    
    x0 = pts.flatten()
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        if res.success or res.fun < float('inf'):
            pts_res = _normalize(res.x.reshape(-1, 3))
            return pts_res, res.fun
    except TimeoutError:
        pass
    except Exception:
        pass
    return pts, _energy_and_grad_cartesian(pts)[0]

def _polish_pass(pts, time_budget=30.0):
    """High-precision polish with tighter tolerances."""
    start_time = time.time()
    remaining = time_budget - (time.time() - start_time)
    if remaining < 5:
        return pts
    
    pts, e = _lbfgs_sphere(pts, maxiter=2000, ftol=1e-16, gtol=1e-14, time_limit=remaining)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Fibonacci spiral
    xyz_init = _fibonacci_points(N)
    best_energy = _energy_and_grad_cartesian(xyz_init)[0]
    best_pts = xyz_init.copy()
    
    # Phase 1: Main optimization with L-BFGS-B
    if time.time() < deadline:
        remaining = deadline - time.time()
        pts1, e1 = _lbfgs_sphere(xyz_init, maxiter=3000, ftol=1e-12, gtol=1e-10, time_limit=remaining * 0.6)
        if e1 < best_energy:
            best_energy = e1
            best_pts = pts1
    
    # Phase 2: Try a second restart if time permits
    if time.time() < deadline - 20:
        np.random.seed(123)
        vecs = np.random.normal(size=(N, 3))
        vecs = _normalize(vecs)
        remaining = deadline - time.time() - 20
        pts2, e2 = _lbfgs_sphere(vecs, maxiter=2000, ftol=1e-12, gtol=1e-10, time_limit=remaining)
        if e2 < best_energy:
            best_energy = e2
            best_pts = pts2
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        best_pts = _polish_pass(best_pts, time_budget=remaining)
        best_energy = _energy_and_grad_cartesian(best_pts)[0]
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
