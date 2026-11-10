import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Initial guess: regular polygon
    # Parameters: 20 radii
    x0 = np.ones(n)
    
    def get_polygon(radii):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    def is_convex(polygon):
        # Check cross products of consecutive edges
        # For CCW polygon, all cross products should be positive (or non-negative)
        # Strict convexity: > 0
        n = len(polygon)
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross <= 1e-9: # Allow small numerical errors, but require strict turn
                return False
        return True
        
    def vertex_score(v, polygon):
        # Calculate distances from v to all other vertices
        diffs = polygon - v
        dists = np.linalg.norm(diffs, axis=1)
        # Exclude self
        idx = np.argmin(dists)
        dists = np.delete(dists, idx)
        dists = np.sort(dists)
        
        # Check min distance
        if dists[0] < 0.05:
            return 0.0
            
        # Find k minimizing d[k+1] - d[k]
        gaps = np.diff(dists)
        k = np.argmin(gaps)
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
            
        # Calculate raw score
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw) if raw > 0 else 0.0

    def objective(radii):
        polygon = get_polygon(radii)
        
        # Normalization step as per contract
        centroid = np.mean(polygon, axis=0)
        polygon_norm = polygon - centroid
        dists_to_centroid = np.linalg.norm(polygon_norm, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
            
        polygon_norm = polygon_norm / mean_dist
        
        # Check convexity
        if not is_convex(polygon_norm):
            return 0.0
            
        # Check repeated points
        # Already handled by convexity check mostly, but let's be safe
        # Actually, is_convex handles turns. Repeated points would cause zero length edge or 180 deg turn.
        
        # Calculate min vertex score
        min_score = 1.0
        for i in range(n):
            s = vertex_score(polygon_norm[i], polygon_norm)
            if s < min_score:
                min_score = s
            if min_score == 0.0:
                break
                
        # We want to maximize min_score, so minimize -min_score
        # Add penalty if convexity fails (already returned 0, which is min -1.0? No, return 0 means score 0.
        # Minimize 0 is better than minimizing -1? No, we want high score.
        # If score is 0, objective is 0. If score is 0.5, objective is -0.5.
        # Minimizer will try to get to -1.0 (score 1.0).
        # So returning 0 for invalid is correct (worse than -0.5).
        
        return -min_score

    # Optimization
    best_score = 0.0
    best_radii = x0
    
    # Initial evaluation
    init_obj = objective(x0)
    if -init_obj > best_score:
        best_score = -init_obj
        best_radii = x0
        
    # Run Nelder-Mead
    # Use maxiter and time limit
    max_iter = 2000
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        res = minimize(objective, best_radii, method='Nelder-Mead', 
                       options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success:
            new_score = -res.fun
            if new_score > best_score:
                best_score = new_score
                best_radii = res.x
            else:
                # If no improvement, maybe break or try random restart?
                # For now, continue with small steps
                pass
        else:
            break
            
    # Final check and return
    final_polygon = get_polygon(best_radii)
    
    # Normalize
    centroid = np.mean(final_polygon, axis=0)
    final_polygon = final_polygon - centroid
    dists_to_centroid = np.linalg.norm(final_polygon, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist > 0.01:
        final_polygon = final_polygon / mean_dist
    else:
        # Fallback to regular polygon if normalization fails
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return final_polygon
