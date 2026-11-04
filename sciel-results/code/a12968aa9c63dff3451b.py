import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    """
    points = np.asarray(points)
    n = len(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    min_area = float('inf')
    # Vectorized calculation for all triples
    # Indices for i < j < k
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    idx_i = i[mask]
    idx_j = j[mask]
    idx_k = k[mask]
    
    p_i = points[idx_i]
    p_j = points[idx_j]
    p_k = points[idx_k]
    
    # Area = 0.5 * |(p_j - p_i) x (p_k - p_i)|
    # Cross product in 2D: x1*y2 - y1*x2
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    if not np.isfinite(min_area) or min_area < 1e-12:
        return 0.0
        
    return - (min_area / hull_area)

def _regular_polygon(n, start_angle=0):
    # Normalize area to 1
    # Area = 0.5 * n * R^2 * sin(2*pi/n) = 1
    # R^2 = 2 / (n * sin(2*pi/n))
    R = np.sqrt(2.0 / (n * np.sin(2.0 * np.pi / n)))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + start_angle
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(13))
    
    # 2. Regular 13-gon rotated
    candidates.append(_regular_polygon(13, np.pi/13))
    
    # 3. A "square-ish" configuration: 4 corners, 4 midpoints, 4 inner, 1 center?
    # Let's try a 3x3 grid scaled + 4 extra points?
    # Maybe just rely on local optimization from regular polygon.
    
    best_score = -np.inf
    best_points = candidates[0]
    
    for init_points in candidates:
        if time.time() - start_time > deadline:
            break
            
        # Local optimization
        # Flatten points
        x0 = init_points.flatten()
        
        # Define objective for minimize
        def objective(x):
            if time.time() - start_time > deadline:
                return 0.0
            pts = x.reshape(-1, 2)
            return _score(pts)
            
        try:
            # Nelder-Mead is robust for non-smooth functions
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result.success or result.fun < best_score:
                best_points = result.x.reshape(-1, 2)
                best_score = result.fun
                # Note: best_score is negative, so lower is better
        except Exception:
            pass
            
    # If no optimization succeeded, return the best initial candidate
    if best_score == -np.inf:
        best_points = candidates[0]
        
    # Final normalization is not strictly necessary for the interface 
    # as the evaluator calculates the score from the returned points,
    # but the points should be valid.
    
    return best_points
