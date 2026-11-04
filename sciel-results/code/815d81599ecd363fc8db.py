"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns a large negative number if invalid (e.g., collinear points).
    """
    n = len(points)
    if n < 3:
        return -1e9
    
    # Check for duplicate points or near-duplicates
    # Simple check: min distance between any pair
    # For N=14, this is fast
    min_dist_sq = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d_sq = dx*dx + dy*dy
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
    if min_dist_sq < 1e-10:
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9
        
    # Compute Min Triangle Area
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                dx1 = pj[0] - pi[0]
                dy1 = pj[1] - pi[1]
                dx2 = pk[0] - pi[0]
                dy2 = pk[1] - pi[1]
                area = 0.5 * abs(dx1*dy2 - dy1*dx2)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return -1e9
                        
    return min_area / hull_area

def _objective(x, lambda_reg=0.01):
    """
    Objective function to minimize.
    x: flattened array of shape (2*N,)
    """
    points = x.reshape(N, 2)
    score_val = _score(points)
    
    if score_val < 0:
        # Invalid configuration
        return 1e9 + lambda_reg * np.sum(x**2)
        
    # Regularization to prevent drift to infinity
    reg = lambda_reg * np.sum(x**2)
    
    # We want to maximize score, so minimize -score
    return -score_val + reg

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = -1e9
    best_points = None
    
    # Initial guesses
    starts = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0
    p_regular = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    starts.append(p_regular.flatten())
    
    # 2. Random points in unit square
    rng = np.random.default_rng(42)
    for _ in range(5):
        p_random = rng.uniform(-1, 1, size=(N, 2))
        starts.append(p_random.flatten())
        
    # 3. Points on circle with random perturbations
    for _ in range(5):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        r = 1.0 + rng.normal(0, 0.2, size=N)
        p_perturbed = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        starts.append(p_perturbed.flatten())

    # Optimize each start
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, x0, method='Powell', options={'maxiter': 200, 'maxfev': 500, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.success or (res.fun < 1e8): # Check if it didn't fail catastrophically
                points = res.x.reshape(N, 2)
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
                    
        except Exception as e:
            pass
            
    # If no valid points found, fallback to regular polygon
    if best_points is None:
        best_points = p_regular
        
    return best_points
# EVOLVE-BLOCK-END
