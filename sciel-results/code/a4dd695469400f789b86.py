# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(polygon):
    """
    Computes the minimum vertex score for the given polygon.
    Returns 0 if invalid (non-convex, etc.).
    """
    n = len(polygon)
    if n != 20:
        return 0.0
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    normalized = centered / mean_dist
    
    # Check strict convexity
    # Compute cross products of consecutive edges
    # Edge i is P_{i+1} - P_i. Cross product Z-component: (x_{i+1}-x_i)(y_{i+2}-y_{i+1}) - (y_{i+1}-y_i)(x_{i+2}-x_{i+1})
    # All must have the same sign and be non-zero.
    
    cross_prods = np.zeros(n)
    for i in range(n):
        p1 = normalized[i]
        p2 = normalized[(i+1) % n]
        p3 = normalized[(i+2) % n]
        e1 = p2 - p1
        e2 = p3 - p2
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_prods[i] = cross
        
    # Check if all cross products are > 0 or all < 0
    if np.any(cross_prods == 0):
        return 0.0
        
    if np.all(cross_prods > 0) or np.all(cross_prods < 0):
        is_convex = True
    else:
        is_convex = False
        
    if not is_convex:
        return 0.0

    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = normalized[i]
        # Distances to all other vertices
        dists = np.linalg.norm(normalized - v, axis=1)
        dists[i] = np.inf # Exclude self
        
        # Sort distances
        sorted_dists = np.sort(dists)
        
        # Check if any distance < 0.05
        if sorted_dists[0] < 0.05:
            min_score = 0.0
            break
            
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = sorted_dists[1:] - sorted_dists[:-1]
        k = np.argmin(diffs)
        
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            min_score = 0.0
            break
            
        # Select 4 distances
        if k == 0:
            sel_dists = sorted_dists[0:4]
        elif k == 17:
            sel_dists = sorted_dists[15:19]
        else:
            sel_dists = sorted_dists[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(sel_dists / D, D / sel_dists)
        raw = np.mean(ratios)
        
        if raw == 0:
            score_v = 0.0
        else:
            score_v = min(1.0, 1.0 / raw)
            
        if score_v < min_score:
            min_score = score_v
            
    return min_score

def _neg_score_with_penalty(polygon_flat):
    """
    Objective function for minimization.
    Minimizes -score + penalty for non-convexity.
    """
    polygon = polygon_flat.reshape(20, 2)
    
    # Penalty for non-convexity
    # We use the same convexity check logic
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 1e6
        
    normalized = centered / mean_dist
    
    cross_prods = np.zeros(20)
    for i in range(20):
        p1 = normalized[i]
        p2 = normalized[(i+1) % 20]
        p3 = normalized[(i+2) % 20]
        e1 = p2 - p1
        e2 = p3 - p2
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_prods[i] = cross
        
    # If not strictly convex, return large penalty
    if not (np.all(cross_prods > 0) or np.all(cross_prods < 0)):
        # Penalty proportional to how much it violates convexity
        # Sum of absolute values of cross products with wrong sign
        if np.any(cross_prods > 0) and np.any(cross_prods < 0):
            penalty = np.sum(np.abs(cross_prods))
        else:
            # All same sign but some zero?
            penalty = 100.0
            
        score = _compute_score(polygon)
        return -score + penalty
        
    score = _compute_score(polygon)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500.0 # Safety margin
    
    # Initial guess: Regular polygon
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    initial_polygon = np.stack([x, y], axis=1)
    
    best_score = _compute_score(initial_polygon)
    best_polygon = initial_polygon.copy()
    
    # Multi-start L-BFGS-B
    np.random.seed(42)
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb the regular polygon
        # Add random noise to angles and radii
        noise_scale = 0.1 + 0.2 * (i / num_restarts)
        angles_pert = angles + np.random.normal(0, noise_scale, n)
        radii_pert = radius * (1 + np.random.normal(0, noise_scale, n))
        
        # Ensure angles are sorted and increasing
        angles_pert = np.sort(angles_pert)
        # Rotate to start at 0? Not strictly necessary, but helps symmetry breaking
        angles_pert = (angles_pert - angles_pert[0]) % (2*np.pi)
        angles_pert = np.sort(angles_pert)
        
        x_pert = radii_pert * np.cos(angles_pert)
        y_pert = radii_pert * np.sin(angles_pert)
        current_polygon = np.stack([x_pert, y_pert], axis=1)
        
        # Flatten for optimizer
        flat_poly = current_polygon.flatten()
        
        # Optimize
        try:
            res = minimize(
                _neg_score_with_penalty,
                flat_poly,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            opt_polygon = res.x.reshape(20, 2)
            opt_score = _compute_score(opt_polygon)
            
            if opt_score > best_score:
                best_score = opt_score
                best_polygon = opt_polygon.copy()
                
        except Exception as e:
            continue
            
    return best_polygon

# EVOLVE-BLOCK-END
