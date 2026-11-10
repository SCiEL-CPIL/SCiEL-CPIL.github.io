import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 590  # Safety margin

    def score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def objective(x):
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists = pdist(p)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1.0
        d_min = np.min(dists)
        return -(d_min / d_max) ** 2

    def fibonacci_sphere(n):
        # Generate n points on a sphere using Fibonacci lattice
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    best_points = None
    best_score = 0.0
    
    # Initial candidate: Fibonacci sphere
    init_pts = fibonacci_sphere(n)
    best_points = init_pts
    best_score = score(init_pts)
    
    # Multi-start local optimization
    # We will try several random restarts and the Fibonacci start
    # Powell is robust for this kind of non-smooth objective
    
    rng = np.random.default_rng(42)
    
    # Number of restarts to attempt
    # Each Powell run might take ~1-5 seconds depending on convergence
    # 600s / 2s ~ 300 runs possible, but let's be conservative and check time
    
    restarts = 0
    max_restarts = 100 # Cap on restarts to avoid infinite loop if Powell is fast
    
    while time.time() - start_time < time_limit - 10:
        if restarts == 0:
            # First iteration uses the Fibonacci sphere
            x0 = init_pts.ravel()
        else:
            # Random restart
            # Scale random points to unit sphere to ensure d_max is reasonable
            x0 = rng.uniform(-1, 1, size=(n, d))
            # Normalize to unit sphere to avoid scale issues in initial step
            norms = np.linalg.norm(x0, axis=1, keepdims=True)
            norms[norms == 0] = 1
            x0 = (x0 / norms).ravel()
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            restarts += 1
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
                
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
        
    return best_points
