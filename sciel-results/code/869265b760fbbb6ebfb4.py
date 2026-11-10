import numpy as np
from scipy.optimize import minimize
import time

def _vertex_score(polygon, i):
    """Calculate the score for vertex i of the polygon."""
    v = polygon[i]
    # Compute distances to all other vertices
    diffs = polygon - v
    diffs = np.delete(diffs, i, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = np.sort(dists)
    
    # Check minimum distance
    if dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(dists)
    min_gap = np.min(gaps)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        ds = dists[0:4]
    elif k == 17:
        ds = dists[15:19]
    else:
        ds = dists[k-1:k+3]
        
    # Calculate raw score
    ratios = np.maximum(ds / D, D / ds)
    raw = np.mean(ratios)
    
    if raw == 0:
        return 1.0
    return min(1.0, 1.0 / raw)

def _total_score(polygon):
    """Calculate the minimum vertex score over all vertices."""
    min_score = 1.0
    for i in range(polygon.shape[0]):
        s = _vertex_score(polygon, i)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                return 0.0
    return min_score

def _is_convex(polygon):
    """Check if polygon is strictly convex (counter-clockwise)."""
    n = polygon.shape[0]
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 1e-9:
            return False
    return True

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Regular polygon as baseline
    R_reg = np.ones(n)
    x_reg = R_reg * np.cos(angles)
    y_reg = R_reg * np.sin(angles)
    polygon_reg = np.stack([x_reg, y_reg], axis=1)
    
    # Normalize baseline
    centroid = np.mean(polygon_reg, axis=0)
    polygon_reg = polygon_reg - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_reg, axis=1))
    if mean_dist > 0.01:
        polygon_reg = polygon_reg / mean_dist
    else:
        return polygon_reg
        
    score_reg = _total_score(polygon_reg)
    
    # Try to optimize
    start_time = time.time()
    time_limit = 600.0
    current_best = score_reg
    best_polygon = polygon_reg.copy()
    
    def objective(params):
        if time.time() - start_time > time_limit - 10:
            # Signal to stop by returning a very bad value
            return 1e9
            
        R = params
        if np.any(R <= 0):
            return 1e9
            
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        # Check convexity
        if not _is_convex(polygon):
            return 1e9
            
        s = _total_score(polygon)
        return -s
        
    # Initial guess: Regular polygon
    x0 = np.ones(n)
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
        
        if res.fun < -current_best:
            R_opt = res.x
            x_opt = R_opt * np.cos(angles)
            y_opt = R_opt * np.sin(angles)
            polygon_opt = np.stack([x_opt, y_opt], axis=1)
            
            # Verify convexity again
            if _is_convex(polygon_opt):
                # Normalize
                centroid_opt = np.mean(polygon_opt, axis=0)
                polygon_opt = polygon_opt - centroid_opt
                mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                if mean_dist_opt > 0.01:
                    polygon_opt = polygon_opt / mean_dist_opt
                    score_opt = _total_score(polygon_opt)
                    if score_opt > current_best:
                        best_polygon = polygon_opt
                        current_best = score_opt
    except Exception:
        pass
        
    return best_polygon

# EVOLVE-BLOCK-END
