import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _log_energy(theta_phi):
    """
    Calculate the logarithmic energy given theta and phi arrays.
    theta_phi is a flat array of length 2N.
    """
    theta = theta_phi[0:N]
    phi = theta_phi[N:2*N]
    
    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    
    # Calculate distances
    # dist[i, j] = |points[i] - points[j]|
    # We want sum_{i<j} log(1/dist)
    # dist^2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
    # = 2 - 2 (xi*xj + yi*yj + zi*zj)
    # So dist = sqrt(2 - 2 dot)
    
    dots = points @ points.T
    dist_sq = 2.0 - 2.0 * dots
    # Avoid zero distance
    dist_sq[dist_sq < 1e-12] = 1e-12
    dist = np.sqrt(dist_sq)
    
    # Mask upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_inv_dist = np.log(1.0 / dist)
    energy = math.fsum(log_inv_dist[mask].flatten())
    
    return energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    theta = np.arccos(z)
    
    initial_theta_phi = np.concatenate([theta, phi])
    
    best_energy = _log_energy(initial_theta_phi)
    best_theta_phi = initial_theta_phi.copy()
    
    # Optimization
    # Use L-BFGS-B
    bounds = [(1e-6, np.pi - 1e-6)] * N + [(0.0, 2.0 * np.pi)] * N
    
    try:
        res = minimize(_log_energy, initial_theta_phi, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if time.time() > deadline:
            # If we are close to deadline, return the best so far
            pass
        else:
            if res.success and res.fun < best_energy:
                best_energy = res.fun
                best_theta_phi = res.x
    except Exception:
        pass
    
    # If we have time left, do a second pass with a different initial guess or higher precision
    if time.time() < deadline - 10.0:
        # Try a different initial guess: random points
        np.random.seed(42)
        random_theta = np.random.uniform(1e-6, np.pi - 1e-6, N)
        random_phi = np.random.uniform(0.0, 2.0 * np.pi, N)
        random_theta_phi = np.concatenate([random_theta, random_phi])
        
        try:
            res2 = minimize(_log_energy, random_theta_phi, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if time.time() > deadline:
                pass
            else:
                if res2.success and res2.fun < best_energy:
                    best_energy = res2.fun
                    best_theta_phi = res2.x
        except Exception:
            pass
    
    # Convert best_theta_phi to Cartesian
    theta = best_theta_phi[0:N]
    phi = best_theta_phi[N:2*N]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points
