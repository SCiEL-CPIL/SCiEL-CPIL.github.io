# EVOLVE-BLOCK-START
import numpy as np
import time

def _vertex_score(v, points):
    """
    Calculate the score for a single vertex v given all points.
    Returns 0 if invalid (distances too small, etc).
    """
    # Distances to all other points
    diffs = points - v
    dists = np.linalg.norm(diffs, axis=1)
    
    # Exclude self (distance 0)
    # v is at index i, but we pass points including v? 
    # Better to pass indices or exclude.
    # Let's assume points is (20,2) and v is points[i].
    # We need distances to the other 19.
    
    # In the main loop, we will handle this.
    pass

def _calc_min_score(polygon):
    n = len(polygon)
    min_score = 1.0
    for i in range(n):
        v = polygon[i]
        # Distances to all other points
        dists = np.linalg.norm(polygon - v, axis=1)
        # Remove self distance (index i)
        # dists[i] is 0
        other_dists = np.delete(dists, i)
        
        # Check if any < 0.05
        if np.any(other_dists < 0.05):
            return 0.0
            
        # Sort
        sorted_dists = np.sort(other_dists)
        
        # Find first index minimizing diff
        # d_{k+1} - d_k
        diffs = np.diff(sorted_dists)
        k = np.argmin(diffs)
        
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            selected = sorted_dists[0:4]
        elif k == 17: # n-2 = 18, so max k is 18? 
            # diffs has length 18. indices 0..17.
            # If k=17, selected d_15..d_18
            selected = sorted_dists[15:19]
        else:
            selected = sorted_dists[k-1:k+3]
            
        # Calculate raw
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Optimization
    start_time = time.time()
    deadline = 580.0 # Safety margin
    
    best_polygon = polygon.copy()
    best_score = _calc_min_score(best_polygon)
    
    # Simple local search: random perturbations + hill climbing
    # Or use scipy if available and fast enough
    # Given 600s, we can do many steps.
    
    # Let's try a simple gradient-free local search
    # Step size adaptive
    step_size = 0.01
    num_steps = 10000
    
    current_polygon = polygon.copy()
    current_score = best_score
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
            
        # Perturb
        noise = np.random.normal(0, step_size, size=(n, 2))
        # Ensure convexity? 
        # Random perturbation might break convexity.
        # We can project back or just accept if score improves and valid.
        
        new_polygon = current_polygon + noise
        
        # Check validity (convexity)
        # Simple check: cross products of consecutive edges all same sign
        valid = True
        for i in range(n):
            p0 = new_polygon[i]
            p1 = new_polygon[(i+1)%n]
            p2 = new_polygon[(i+2)%n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(cross) < 1e-9:
                valid = False
                break
            if i == 0:
                sign = 1 if cross > 0 else -1
            else:
                if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    valid = False
                    break
        
        if not valid:
            # Reduce step size if invalid often?
            if np.random.rand() < 0.1:
                step_size *= 0.9
            continue
            
        new_score = _calc_min_score(new_polygon)
        
        if new_score > current_score:
            current_polygon = new_polygon
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_polygon = current_polygon.copy()
            # Increase step size slightly?
            step_size *= 1.05
        else:
            # Decrease step size
            step_size *= 0.95
            
        if best_score >= 1.0:
            break
            
    return best_polygon

# EVOLVE-BLOCK-END
