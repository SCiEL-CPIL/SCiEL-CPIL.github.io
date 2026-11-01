"""
Optimizes 14 points on the unit circle to maximize the minimum triangle area
relative to the convex hull area.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area == 0:
        return 0.0
        
    # Calculate all triangle areas
    n = len(points)
    min_area = float('inf')
    # Optimization: We can compute areas efficiently
    # A(i,j,k) = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if area < min_area:
                    min_area = area
                    if min_area == 0:
                        return 0.0
                        
    return min_area / hull_area

def objective(log_gaps):
    """
    Objective function for optimization.
    log_gaps: array of length N.
    """
    # Softmax-like normalization to ensure positive gaps summing to 2*pi
    # Use exp(log_gaps) to ensure positivity
    weights = np.exp(log_gaps - np.max(log_gaps)) # Stability
    gaps = 2 * np.pi * weights / np.sum(weights)
    
    # Construct angles
    angles = np.cumsum(gaps)
    
    # Construct points on unit circle
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Compute score
    score = compute_score(points)
    
    # We want to maximize score, so minimize negative score
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initial guess: regular polygon (equal gaps)
    # log_gaps = [0, 0, ..., 0]
    x0 = np.zeros(N)
    
    best_score = 0.0
    best_points = None
    
    # Run Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
        
        # Check time
        if time.time() > deadline:
             # If timeout, we might have a partial result, but let's just use what we have if valid
             pass
             
        if res.fun < 0: # res.fun is -score
             score = -res.fun
             if score > best_score:
                 best_score = score
                 # Reconstruct points from res.x
                 weights = np.exp(res.x - np.max(res.x))
                 gaps = 2 * np.pi * weights / np.sum(weights)
                 angles = np.cumsum(gaps)
                 best_points = np.column_stack((np.cos(angles), np.sin(angles)))
                 
    except Exception:
        pass
        
    # If Nelder-Mead didn't work well or timed out badly, try a few random restarts if time permits
    if time.time() < deadline - 20.0:
        rng = np.random.RandomState(42)
        for _ in range(10): # 10 restarts
            if time.time() > deadline - 10.0:
                break
            x0_rand = rng.normal(0, 0.5, N)
            try:
                res = minimize(objective, x0_rand, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
                if res.fun < 0:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        weights = np.exp(res.x - np.max(res.x))
                        gaps = 2 * np.pi * weights / np.sum(weights)
                        angles = np.cumsum(gaps)
                        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
            except Exception:
                pass
                
    # If no valid points found (should not happen), return regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
