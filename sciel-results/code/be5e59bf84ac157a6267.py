import numpy as np
from scipy.optimize import minimize
import math

N = 100

def _log_energy(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|)
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist_safe = np.maximum(dist, 1e-15)
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    return -np.sum(np.log(d_upper))

def _penalty_term(pts, lambda_pen=1000.0):
    """
    Penalty for deviating from unit sphere.
    """
    norms_sq = np.sum(pts**2, axis=1)
    return lambda_pen * np.sum((norms_sq - 1.0)**2)

def _objective(pts_flat, lambda_pen=1000.0):
    """
    Objective function for scipy.optimize.minimize.
    Takes flat array, reshapes to (N, 3).
    """
    pts = pts_flat.reshape(N, 3)
    return _log_energy(pts) + _penalty_term(pts, lambda_pen)

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    # Start from a Fibonacci spiral (good initial guess)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = np.inf
    best_pts = pts_init.copy()
    
    n_restarts = 50
    lambda_pen = 1000.0
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = pts_init.copy()
        elif restart == 1:
            # Random rotation of Fibonacci
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            x0 = pts_init @ rot.T
        else:
            # Random perturbation or random init
            if restart % 2 == 0:
                # Perturbed Fibonacci
                noise = np.random.normal(0, 0.1, size=(N, 3))
                x0 = pts_init + noise
            else:
                # Random points on sphere
                x0 = np.random.normal(size=(N, 3))
                
        x0 = _project_to_sphere(x0)
        x0_flat = x0.flatten()
        
        try:
            res = minimize(_objective, x0_flat, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                res_pts = _project_to_sphere(res.x.reshape(N, 3))
                res_energy = _log_energy(res_pts)
                if res_energy < best_energy:
                    best_energy = res_energy
                    best_pts = res_pts
        except Exception as e:
            pass

    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    # ... (same as before)
    
    return best_pts
