import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    # Check for valid points
    if not np.all(np.isfinite(points)):
        return 0.0
        
    n = points.shape[0]
    
    # Calculate minimum triangle area
    min_area = np.inf
    # Loop over all triples
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Cross product magnitude gives 2 * area
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 0:
        return 0.0

    return min_area / hull_area

def _negative_score(flat_points):
    points = flat_points.reshape(N, 2)
    return -_calculate_score(points)

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # 1. Initialize with Regular Polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = _calculate_score(init_points)
    best_points = init_points.copy()
    
    # 2. Optimize using Powell
    # We optimize the 2N coordinates.
    # Powell is chosen because it doesn't require gradients and handles non-smoothness (min area) reasonably well.
    try:
        res = minimize(
            _negative_score, 
            init_points.flatten(), 
            method='Powell', 
            options={'maxiter': 1000, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
        )
        
        if time.time() - start_time > timeout:
            return best_points
            
        if res.x is not None:
            candidate_points = res.x.reshape(N, 2)
            cand_score = _calculate_score(candidate_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate_points
                
    except Exception:
        pass

    # 3. If time remains, try a second optimization starting from the best point found
    # with a different method (Nelder-Mead) to potentially escape local optima.
    if time.time() - start_time < timeout - 30:
        try:
            res2 = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'disp': False}
            )
            if res2.x is not None:
                candidate_points2 = res2.x.reshape(N, 2)
                cand_score2 = _calculate_score(candidate_points2)
                if cand_score2 > best_score:
                    best_score = cand_score2
                    best_points = candidate_points2
        except Exception:
            pass

    return best_points
