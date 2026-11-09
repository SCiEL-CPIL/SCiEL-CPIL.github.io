# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(v, points):
    """
    Calculate the score for a single vertex v in the polygon.
    """
    # Distances from v to all other points
    diffs = points - v
    dists = np.linalg.norm(diffs, axis=1)
    
    # Exclude the vertex itself (distance 0)
    # The vertex is at index i, so dists[i] is 0.
    # We need the 19 other distances.
    mask = np.ones(len(dists), dtype=bool)
    # Find the index of v in points to mask it out properly if not passed as index
    # But here v is one of the points. We can just sort and remove the 0.
    # Since floating point, dists might not be exactly 0.
    # Better to pass index.
    
def _vertex_score_idx(i, points):
    """
    Calculate the score for vertex at index i.
    """
    v = points[i]
    diffs = points - v
    dists = np.linalg.norm(diffs, axis=1)
    # Remove the distance to self
    dists = np.delete(dists, i)
    dists = np.sort(dists)
    
    # Check for small distances
    if dists[0] < 0.05:
        return 0.0
        
    # Find k minimizing d_{k+1} - d_k
    # d_0 ... d_18
    gaps = np.diff(dists)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        sel = dists[0:4]
    elif k == 17: # 19 distances, indices 0..18. Gap between k and k+1. Max k is 17 (gap 17-18)
        sel = dists[15:19]
    else:
        sel = dists[k-1:k+3]
        
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(sel / D, D / sel)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(points):
    """
    Calculate the minimum vertex score for the polygon.
    """
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points = points / mean_dist
    
    # Check convexity
    # Compute cross products of consecutive edges
    # p_i, p_{i+1}, p_{i+2}
    # (p_{i+1} - p_i) x (p_{i+2} - p_{i+1})
    n = len(points)
    cross_prods = np.zeros(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross_prods[i] = e1[0] * e2[1] - e1[1] * e2[0]
        
    # All cross products must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-9):
        return 0.0
        
    if not np.all(cross_prods > 0) and not np.all(cross_prods < 0):
        return 0.0
        
    # Calculate vertex scores
    min_score = 1.0
    for i in range(n):
        s = _vertex_score_idx(i, points)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score

def _neg_min_score(radii):
    """
    Objective function for optimization.
    radii: array of 20 values.
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.stack([x, y], axis=1)
    
    score = _score_polygon(points)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial guess: regular polygon
    radii = np.ones(n)
    
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    best_radii = radii
    best_score = _score_polygon(np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1))
    
    # Optimization
    # We use Nelder-Mead as it doesn't require gradients and is robust for small N.
    # We try a few random restarts to escape local minima.
    
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - safety_margin:
            break
            
        if restart == 0:
            current_radii = best_radii
        else:
            # Perturb the best radii or random start
            # Random start in [0.8, 1.2]
            current_radii = best_radii + np.random.uniform(-0.2, 0.2, n)
            # Ensure radii are positive
            current_radii = np.maximum(current_radii, 0.5)
            
        try:
            res = minimize(
                _neg_min_score, 
                current_radii, 
                method='Nelder-Mead', 
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000}
            )
            
            if res.success or res.fun < -best_score:
                # Check if this is valid and better
                candidate_points = np.stack([res.x * np.cos(angles), res.x * np.sin(angles)], axis=1)
                candidate_score = _score_polygon(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_radii = res.x
                    
        except Exception:
            pass
            
    # Final construction
    x = best_radii * np.cos(angles)
    y = best_radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon

# EVOLVE-BLOCK-END
