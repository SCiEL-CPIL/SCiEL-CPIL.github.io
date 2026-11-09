import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    points_list = []
    
    # Generate all vectors with two non-zero entries being +/-1
    # These have squared norm 2.
    for i in range(d):
        for j in range(i + 1, d):
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points_list.append(v1)
            
            v2 = -v1
            points_list.append(v2)
            
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = 1
            v3[j] = -1
            points_list.append(v3)
            
            v4 = -v3
            points_list.append(v4)
            
    n_points = len(points_list)
    points = np.array(points_list)
    
    # The constraint is: min_dist_sq >= max_norm_sq
    # For this set, max_norm_sq = 2.
    # So we need dist_sq >= 2 for all pairs.
    # dist_sq(u, v) = ||u||^2 + ||v||^2 - 2 u.v = 2 + 2 - 2 u.v = 4 - 2 u.v
    # We need 4 - 2 u.v >= 2  =>  2 >= 2 u.v  =>  u.v <= 1.
    # Conflict if u.v > 1.
    # Since u, v are integers, u.v is integer.
    # Max u.v is 2 (if u=v).
    # So the only conflict is u=v.
    # Since all points in points_list are distinct, there are NO conflicts.
    # We can return all points.
    
    # Verification step to be safe
    if len(points) < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    max_norm = np.max(np.sum(points**2, axis=1))
    
    # Check min dist squared
    min_dist = float('inf')
    # For efficiency, we can vectorize or just loop. 
    # 220 points -> 24200 pairs. Very fast.
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d2 = np.dot(points[i] - points[j], points[i] - points[j])
            if d2 < min_dist:
                min_dist = d2
                 
    if min_dist < max_norm:
        # Fallback: This should not happen for this specific set, 
        # but if it does, we need a valid subset.
        # We can greedily pick points that satisfy the condition.
        valid_points = []
        for i in range(len(points)):
            # Check if points[i] is compatible with all already selected
            compatible = True
            for j in range(len(valid_points)):
                d2 = np.dot(points[i] - valid_points[j], points[i] - valid_points[j])
                # Need d2 >= max(max_norm_of_selected, norm_sq_of_i)
                # Since all norms are 2, we need d2 >= 2.
                if d2 < 2:
                    compatible = False
                    break
            if compatible:
                valid_points.append(points[i])
        
        if len(valid_points) >= 2:
             return np.array(valid_points, dtype=np.int64)
             
        # Absolute fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    return points
