import numpy as np
import time
from scipy.optimize import minimize

def compute_scores(polygon):
    # polygon: (20, 2)
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self
        dists[i] = np.inf
        # Sort
        dists = np.sort(dists)
        # d_0 ... d_18
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists)
        k = np.argmin(diffs)
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17: # n-2 = 18, so max k is 18? 
            # dists has 19 elements. indices 0..18.
            # diffs has 18 elements. indices 0..17.
            # k is in 0..17.
            # if k=17, take d_15..d_18
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3] # k-1, k, k+1, k+2
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def objective(params):
    # params: 40 floats: 20 angles, 20 radii
    angles = params[:20]
    radii = params[20:]
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.column_stack((x, y))
    
    # Convexity check
    # Calculate cross products of consecutive edges
    # If any cross product is negative (assuming CCW), penalty
    # To ensure robustness, we can just check if points are in convex position
    # Simple check: all cross products of (v_{i+1}-v_i) x (v_{i+2}-v_{i+1}) should have same sign
    
    # Normalize for score calculation? 
    # The scoring function in the task is scale/translation invariant.
    # But my compute_scores assumes the input is the raw polygon.
    # The task says: "The polygon is normalized: subtract the centroid, divide by the mean distance to it..."
    # So I must normalize before scoring.
    
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
    if mean_dist <= 0.01:
        return 1000.0 # Invalid
    polygon_norm = polygon_norm / mean_dist
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    
    # Convexity penalty
    # Compute cross products
    v0 = polygon_norm[0]
    v1 = polygon_norm[1]
    v2 = polygon_norm[2]
    # ...
    # Efficient convexity check for N=20
    # Cross product of edge i and edge i+1
    # (v_{i+1} - v_i) x (v_{i+2} - v_{i+1})
    # z-component: (x1-x0)(y2-y1) - (y1-y0)(x2-x1)
    
    # Vectorized
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # For a convex CCW polygon, all cross should be > 0
    # If any are <= 0, it's not strictly convex or self-intersecting
    # Penalty: sum of negative cross products squared
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    return -min_score + 10.0 * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0 - 10.0 # Safety margin
    
    n = 20
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n)
    x0 = np.concatenate([angles, radii])
    
    best_params = x0
    best_score = -1e9
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if time.time() > deadline:
             # Should not happen if maxiter is low, but check
             pass
             
        # Evaluate best found
        # res.x is the best
        # But we might have timed out? No, minimize handles it.
        
        # Verify the result
        angles_opt = res.x[:20]
        radii_opt = res.x[20:]
        
        x = radii_opt * np.cos(angles_opt)
        y = radii_opt * np.sin(angles_opt)
        polygon = np.column_stack((x, y))
        
        # Normalize
        centroid = np.mean(polygon, axis=0)
        polygon_norm = polygon - centroid
        mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
        if mean_dist <= 0.01:
             return np.column_stack((np.cos(angles), np.sin(angles))) # Fallback to regular
        
        polygon_norm = polygon_norm / mean_dist
        
        scores = compute_scores(polygon_norm)
        min_score = np.min(scores)
        
        # Check convexity strictly
        v_curr = polygon_norm
        v_next = np.roll(polygon_norm, -1, axis=0)
        v_next2 = np.roll(polygon_norm, -2, axis=0)
        e1 = v_next - v_curr
        e2 = v_next2 - v_next
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        
        if np.any(cross <= 1e-6): # Not strictly convex
             # Fallback to regular
             return np.column_stack((np.cos(angles), np.sin(angles)))
             
        return polygon_norm
    except Exception:
        # Fallback to regular
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
