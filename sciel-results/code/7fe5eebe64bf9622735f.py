import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes safety margin

    # Objective function to minimize: -softmin(distances) + penalty(sphere)
    def objective(params, t=50.0, lambda_pen=10.0):
        points = params.reshape(n, d)
        
        # Calculate pairwise distances
        dists = pdist(points)
        
        # Soft min approximation
        # Using log-sum-exp trick for stability
        # softmin = -1/t * log(sum(exp(-t * dists)))
        # To avoid overflow/underflow, subtract min
        min_d = np.min(dists)
        log_sum_exp = min_d + np.log(np.sum(np.exp(-t * (dists - min_d))))
        soft_min = -log_sum_exp / t
        
        # Penalty for not being on unit sphere
        norms = np.linalg.norm(points, axis=1)
        pen = np.sum((norms - 1.0)**2)
        
        return -soft_min + lambda_pen * pen

    def gradient(params, t=50.0, lambda_pen=10.0):
        # Numerical gradient is safer and fast enough for 42 variables
        # Or we can derive it, but numerical is robust for this size
        # Let's use numerical for simplicity and reliability within time limit
        # Actually, minimize can compute numerical gradient itself if we don't provide one.
        # Providing it might be faster if we derive it, but let's stick to standard minimize first.
        pass

    # Initialize candidates
    candidates = []
    
    # 1. Fibonacci Sphere
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    fib_points = np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))
    candidates.append(fib_points.flatten())

    # 2. Random points on sphere with noise
    np.random.seed(42)
    for i in range(20):
        # Generate random points on sphere
        vecs = np.random.randn(n, d)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        pts = vecs / norms
        # Add small noise
        pts += 0.05 * np.random.randn(n, d)
        candidates.append(pts.flatten())

    best_score = -np.inf
    best_points = None

    # Optimize each candidate
    for i, init in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(
                objective, 
                init, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if res.success or res.fun < best_score:
                # Evaluate actual min distance for this result
                pts = res.x.reshape(n, d)
                dists = pdist(pts)
                min_d = np.min(dists)
                max_d = np.max(dists)
                if max_d > 0:
                    ratio = (min_d / max_d) ** 2
                else:
                    ratio = 0
                    
                if ratio > best_score:
                    best_score = ratio
                    best_points = pts
        except Exception as e:
            continue

    # If we have time, do a second pass with higher precision on the best point
    if best_points is not None and time.time() < deadline - 60:
        # Try to polish with a tighter tolerance or different method
        # We can also try increasing 't' in the objective for the polish
        def objective_polish(params, t=100.0, lambda_pen=100.0):
            points = params.reshape(n, d)
            dists = pdist(points)
            min_d = np.min(dists)
            log_sum_exp = min_d + np.log(np.sum(np.exp(-t * (dists - min_d))))
            soft_min = -log_sum_exp / t
            norms = np.linalg.norm(points, axis=1)
            pen = np.sum((norms - 1.0)**2)
            return -soft_min + lambda_pen * pen

        try:
            res2 = minimize(
                objective_polish, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
            )
            pts2 = res2.x.reshape(n, d)
            dists2 = pdist(pts2)
            min_d2 = np.min(dists2)
            max_d2 = np.max(dists2)
            if max_d2 > 0:
                ratio2 = (min_d2 / max_d2) ** 2
            else:
                ratio2 = 0
                
            if ratio2 > best_score:
                best_points = pts2
                best_score = ratio2
        except Exception:
            pass

    if best_points is None:
        # Fallback to Fibonacci sphere if optimization failed
        best_points = fib_points

    # Normalize to have max distance 1 (optional, but ensures clean output)
    # The ratio is invariant to scale, but scaling to max_dist=1 is standard
    dists_final = pdist(best_points)
    max_d_final = np.max(dists_final)
    if max_d_final > 0:
        best_points = best_points / max_d_final

    return best_points
