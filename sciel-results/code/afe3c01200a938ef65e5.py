import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Time limit
    start_time = time.time()
    time_limit = 550  # Safety margin

    # 1. Initialization: Fibonacci Sphere
    # This provides a good initial spread
    y = np.cos(np.arccos(1 - 2 * (np.arange(n) + 0.5) / n))
    p = np.sqrt(1 - y**2)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    
    points = np.column_stack((
        p * np.cos(theta),
        p * np.sin(theta),
        y
    ))
    
    # 2. Optimization
    # Objective: Maximize the minimum pairwise distance
    # We minimize the negative of the minimum distance
    
    def objective(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        return -np.min(dists)
        
    def callback(xk):
        nonlocal start_time
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")

    try:
        # Use Powell method which is derivative-free and robust
        result = minimize(
            objective, 
            points.flatten(), 
            method='Powell',
            callback=callback,
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.success or result.x is not None:
            points = result.x.reshape(n, d)
            
    except RuntimeError:
        pass # Return best found so far
    
    # Verify finite
    if not np.all(np.isfinite(points)):
        # Fallback to initial
        y = np.cos(np.arccos(1 - 2 * (np.arange(n) + 0.5) / n))
        p = np.sqrt(1 - y**2)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        points = np.column_stack((
            p * np.cos(theta),
            p * np.sin(theta),
            y
        ))

    return points
