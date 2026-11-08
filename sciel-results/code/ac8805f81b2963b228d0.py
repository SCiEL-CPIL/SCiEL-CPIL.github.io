import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _min_triangle_area(points):
    n = points.shape[0]
    min_area = np.inf
    # Vectorized approach for speed
    # points: (N, 2)
    # We need cross product of (p_j - p_i) and (p_k - p_i) for all i<j<k
    # Simple loop is fast enough for N=14
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 1e-10 # Fallback

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return -1e10
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initial guesses
    # 1. Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Generate a few perturbations
    rng = np.random.default_rng(42)
    initial_guesses = [base_points]
    for _ in range(10):
        noise = rng.normal(0, 0.1, (N, 2))
        initial_guesses.append(base_points + noise)
        
    # Also try a "square-like" distribution? No, regular polygon is best for max min area.
    # Maybe try points on a circle with slightly varied radii?
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        x0 = guess.flatten()
        try:
            res = minimize(_negative_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if time.time() > deadline:
                break
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(-1, 2)
        except Exception:
            pass

    if best_points is None:
        return base_points
        
    return best_points
