import numpy as np
from scipy.optimize import minimize
import time

def _compute_vertex_score(polygon, v_idx):
    """
    Compute the score for a single vertex v_idx in the polygon.
    polygon: (20, 2) array.
    """
    v = polygon[v_idx]
    # Distances to all other vertices
    # Exclude v itself
    mask = np.ones(len(polygon), dtype=bool)
    mask[v_idx] = False
    other_pts = polygon[mask]
    
    # Euclidean distances
    diffs = other_pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Sort distances
    dists = np.sort(dists)
    
    # n = 20, so 19 other vertices
    # d_0 ... d_18
    if len(dists) != 19:
        return 0.0
        
    # Check d_0 < 0.05
    if dists[0] < 0.05:
        return 0.0
        
    # Find k minimizing d_{k+1} - d_k
    gaps = np.diff(dists)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        d_sel = dists[0:4]
    elif k == 17: # k+1 = 18, so k=17 is the last gap
        d_sel = dists[15:19]
    else:
        d_sel = dists[k-1:k+3]
        
    # Calculate raw score
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    if raw < 1e-15:
        return 1.0
        
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score for the polygon.
    """
    scores = []
    for i in range(len(polygon)):
        s = _compute_vertex_score(polygon, i)
        scores.append(s)
    return min(scores)

def _evaluate_params(theta_vars):
    """
    theta_vars: array of 9 angles (theta_1 ... theta_9)
    Returns negative of the min score (to minimize).
    """
    # Construct full angles
    # theta_0 = 0
    # theta_10 = pi
    # theta_{10+i} = 2*pi - theta_{10-i}
    
    thetas = np.zeros(20)
    thetas[0] = 0.0
    thetas[1:10] = theta_vars
    thetas[10] = np.pi
    # Symmetry: theta_{10+i} = 2*pi - theta_{10-i}
    # i=1: theta_11 = 2*pi - theta_9
    # i=2: theta_12 = 2*pi - theta_8
    # ...
    # i=9: theta_19 = 2*pi - theta_1
    
    for i in range(1, 10):
        thetas[10+i] = 2.0 * np.pi - theta_vars[10-i]
        
    # Ensure strict ordering
    # Check if any angle is out of bounds or not strictly increasing
    # theta_vars should be in (0, pi) and sorted
    
    # Check bounds
    if np.any(theta_vars <= 0) or np.any(theta_vars >= np.pi):
        return 1e6 # Large penalty
        
    # Check sorting
    if not np.all(np.diff(theta_vars) > 0):
        return 1e6
        
    # Check against theta_10
    if theta_vars[-1] >= np.pi:
        return 1e6
        
    # Check against theta_0
    if theta_vars[0] <= 0:
        return 1e6
        
    # Construct polygon
    x = np.cos(thetas)
    y = np.sin(thetas)
    polygon = np.stack([x, y], axis=1)
    
    # Normalize (though for unit circle centered at 0, it's already normalized)
    # But let's do it to be safe
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(polygon**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 1e6
    polygon = polygon / mean_dist
    
    # Calculate score
    min_score = _polygon_min_score(polygon)
    
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Initial guess: regular polygon
    # theta_i = 2*pi*i/20
    # theta_1 = 2*pi/20 = pi/10
    # ...
    # theta_9 = 18*pi/20 = 9*pi/10
    
    theta_init = np.linspace(np.pi/10, 9*np.pi/10, 9)
    
    # Bounds for Nelder-Mead
    # theta_i in (0, pi)
    # We need strict inequalities.
    # Use small epsilon
    eps = 1e-5
    bounds = [(eps, np.pi - eps) for _ in range(9)]
    
    # Nelder-Mead doesn't support bounds directly in older SciPy, but we can use penalty or TNC.
    # Let's use 'Nelder-Mead' and handle bounds via penalty or just start inside and hope it stays.
    # Better: use 'L-BFGS-B' if we had gradients. We don't.
    # Use 'Powell' or 'Nelder-Mead' with careful initialization.
    
    # Let's try a few iterations
    start_time = time.time()
    deadline = start_time + 500.0 # Leave 100s for safety
    
    best_result = None
    best_score = -1e9
    best_theta = theta_init.copy()
    
    # We can run multiple restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            theta_start = theta_init
        else:
            # Add noise
            noise = np.random.uniform(-0.1, 0.1, 9)
            theta_start = theta_init + noise
            # Clip to bounds
            theta_start = np.clip(theta_start, eps, np.pi - eps)
            # Sort? No, the order is fixed by index.
            # If noise breaks sorting, the penalty will be high.
            # Let's ensure sorting for the start point
            theta_start = np.sort(theta_start)
            
        res = minimize(_evaluate_params, theta_start, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_theta = res.x
            best_result = res
            
        # Check time
        if time.time() > deadline:
            break
            
    # Construct final polygon from best_theta
    if best_theta is None:
        best_theta = theta_init
        
    # Ensure best_theta is valid
    if not np.all(best_theta > 0) or not np.all(best_theta < np.pi):
        best_theta = theta_init
        
    thetas = np.zeros(20)
    thetas[0] = 0.0
    thetas[1:10] = best_theta
    thetas[10] = np.pi
    for i in range(1, 10):
        thetas[10+i] = 2.0 * np.pi - best_theta[10-i]
        
    x = np.cos(thetas)
    y = np.sin(thetas)
    polygon = np.stack([x, y], axis=1)
    
    # Final normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(polygon**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
        
    return polygon
