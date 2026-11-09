# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(distances):
    """
    Compute the score for a single vertex given sorted distances to other vertices.
    """
    d = np.sort(distances)
    # d has length 19
    if np.any(d < 0.05):
        return 0.0
    
    # Find k minimizing d[k+1] - d[k]
    diffs = d[1:] - d[:-1]
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17:
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    d_sel = d[idxs]
    
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw) if raw > 0 else 0.0

def _evaluate_polygon(polygon):
    """
    Evaluate the polygon: normalize, check convexity, compute min vertex score.
    """
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(polygon, axis=0)
    poly_norm = polygon - centroid
    dists_to_centroid = np.linalg.norm(poly_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    poly_norm = poly_norm / mean_dist
    
    # Check strict convexity
    # 1. No repeated points
    if np.any(np.linalg.norm(np.diff(np.vstack([poly_norm, poly_norm[0]]), axis=0), axis=1) < 1e-6):
        return 0.0
        
    # 2. Every turn nonzero and of one sign
    turns = []
    for i in range(n):
        p0 = poly_norm[i]
        p1 = poly_norm[(i+1)%n]
        p2 = poly_norm[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        turns.append(cross)
    turns = np.array(turns)
    if np.any(np.abs(turns) < 1e-9):
        return 0.0
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return 0.0
        
    # 3. Total turning +-2*pi
    # For a convex polygon, the sum of exterior angles is 2*pi.
    # The cross product check above ensures local convexity.
    # We should verify the total angle change is 2*pi.
    # Actually, for a simple convex polygon, if all turns are same sign and non-zero, 
    # and it closes, the total turn is 2*pi.
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        vi = poly_norm[i]
        dists = np.linalg.norm(poly_norm - vi, axis=1)
        # Exclude self distance
        dists[i] = np.inf
        score = _vertex_score(dists)
        if score < min_score:
            min_score = score
            
    return min_score

def _polygon_from_radii(radii):
    """
    Convert radii to polygon coordinates.
    """
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, keep margin
    
    # Initial guess: regular polygon
    radii_init = np.ones(n)
    
    # Bounds for radii
    bounds = [(0.5, 1.5) for _ in range(n)]
    
    # Objective function for optimizer (minimize negative score)
    def objective(radii):
        polygon = _polygon_from_radii(radii)
        score = _evaluate_polygon(polygon)
        return -score
        
    # Check deadline
    if time.time() > deadline - 10.0:
        return _polygon_from_radii(radii_init)
        
    # Phase 1: Differential Evolution for global search
    # Use a moderate number of iterations to fit within time
    # maxiter and popsize control the work
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-4,
            polish=False # We will polish separately
        )
        best_radii = res.x
    except Exception:
        best_radii = radii_init
        
    # Check deadline
    if time.time() > deadline - 30.0:
        return _polygon_from_radii(best_radii)
        
    # Phase 2: Local Polish using L-BFGS-B
    # Since the score function is non-smooth, L-BFGS-B might struggle.
    # We can try to use a smooth surrogate or just do a fine grid search / Nelder-Mead.
    # Let's use Nelder-Mead for local refinement as it is derivative-free and robust to non-smoothness.
    try:
        res2 = minimize(
            objective, 
            best_radii, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if res2.success and -res2.fun > -objective(best_radii):
            best_radii = res2.x
    except Exception:
        pass
        
    # Check deadline
    if time.time() > deadline - 10.0:
        return _polygon_from_radii(best_radii)
        
    # Final check: ensure the best solution is valid and better than regular polygon
    score_best = _evaluate_polygon(_polygon_from_radii(best_radii))
    score_regular = _evaluate_polygon(_polygon_from_radii(radii_init))
    
    if score_best > score_regular:
        return _polygon_from_radii(best_radii)
    else:
        return _polygon_from_radii(radii_init)

# EVOLVE-BLOCK-END
