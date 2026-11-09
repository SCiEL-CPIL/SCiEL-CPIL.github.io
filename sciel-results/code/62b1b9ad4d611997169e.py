import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _raw_log_energy(pts):
    """Compute the exact logarithmic energy using math.fsum for accuracy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())

def _energy_and_grad(x_flat, lambda_pen=10.0):
    """Compute energy and gradient for L-BFGS-B optimization."""
    x = x_flat.reshape(N, 3)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    log_dists = np.log(dist)
    log_dists[~mask] = 0.0
    E_log = -0.5 * np.sum(log_dists)
    
    norms_sq = np.sum(x * x, axis=1)
    pen = lambda_pen * np.sum((norms_sq - 1.0)**2)
    
    total_energy = E_log + pen
    
    grad_log = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * x
    total_grad = grad_log + grad_pen
    
    return total_energy, total_grad.flatten()

def _project_to_sphere(pts):
    """Project points exactly onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_points(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def construct():
    t_start = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Phase 1: Coarse optimization with L-BFGS-B
    x0 = _fibonacci_points(N)
    best_x = x0.copy()
    best_energy = _raw_log_energy(best_x)
    
    try:
        res1 = minimize(
            _energy_and_grad, 
            x0.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': 500,
                'ftol': 1e-10,
                'gtol': 1e-6,
                'maxls': 50
            }
        )
        
        candidate = _project_to_sphere(res1.x.reshape(N, 3))
        cand_energy = _raw_log_energy(candidate)
        
        if cand_energy < best_energy:
            best_x = candidate
            best_energy = cand_energy
            
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - t_start
    if elapsed > time_limit - 30:
        return best_x
    
    # Phase 2: Fine polish with tighter tolerances
    try:
        res2 = minimize(
            _energy_and_grad, 
            best_x.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': 1000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 100
            }
        )
        
        candidate = _project_to_sphere(res2.x.reshape(N, 3))
        cand_energy = _raw_log_energy(candidate)
        
        if cand_energy < best_energy:
            best_x = candidate
            best_energy = cand_energy
            
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - t_start
    if elapsed > time_limit - 10:
        return best_x
    
    # Phase 3: Final high-precision polish with very tight tolerances
    try:
        res3 = minimize(
            _energy_and_grad, 
            best_x.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': 500,
                'ftol': 1e-16,
                'gtol': 1e-12,
                'maxls': 100
            }
        )
        
        candidate = _project_to_sphere(res3.x.reshape(N, 3))
        cand_energy = _raw_log_energy(candidate)
        
        if cand_energy < best_energy:
            best_x = candidate
            best_energy = cand_energy
            
    except Exception:
        pass
    
    # Final normalization
    best_x = _project_to_sphere(best_x)
    
    return best_x
