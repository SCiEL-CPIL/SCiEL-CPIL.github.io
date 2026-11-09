import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Scoring function
    def score_polygon(radii):
        # Check convexity
        # x = r * cos(theta), y = r * sin(theta)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.stack([x, y], axis=1)
        
        # Check convexity: cross product of (p_{i+1}-p_i) and (p_{i+2}-p_{i+1})
        # All must have the same sign (non-zero)
        edges = np.roll(pts, -1, axis=0) - pts
        cross_products = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # Strictly convex: no repeated points, all cross products same sign and non-zero
        if np.any(np.abs(cross_products) < 1e-12):
            return 0.0
        if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
            return 0.0
            
        # Check no repeated points
        # Distances between consecutive points
        dists_consec = np.linalg.norm(edges, axis=1)
        if np.any(dists_consec < 1e-12):
            return 0.0
            
        # Normalize
        centroid = np.mean(pts, axis=0)
        pts_norm = pts - centroid
        dists_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = np.mean(dists_centroid)
        if mean_dist <= 0.01:
            return 0.0
        pts_norm = pts_norm / mean_dist
        
        # Calculate vertex scores
        min_score = 1.0
        for i in range(n):
            v = pts_norm[i]
            # Distances to others
            diffs = pts_norm - v
            # Exclude self
            diffs[i] = np.array([np.inf, np.inf])
            dists = np.linalg.norm(diffs, axis=1)
            # Sort
            dists_sorted = np.sort(dists)
            
            # Check min distance
            if dists_sorted[0] < 0.05:
                min_score = 0.0
                break
                
            # Find k
            gaps = np.diff(dists_sorted)
            k = np.argmin(gaps)
            D = (dists_sorted[k] + dists_sorted[k+1]) / 2.0
            if D < 0.1:
                min_score = 0.0
                break
                
            # Select 4 distances
            if k == 0:
                sel_dists = dists_sorted[0:4]
            elif k == 17: # n-3? n=20, indices 0..18. 18 is last.
                # If k=17, d_17, d_18 are the pair.
                # d_{k-1}..d_{k+2} would be d_16..d_19 (out of bounds)
                # Rule: "d_15..d_18 if k=17"
                sel_dists = dists_sorted[15:19]
            else:
                # d_{k-1}..d_{k+2}
                sel_dists = dists_sorted[k-1:k+3]
                
            # Calculate raw score
            ratios = np.maximum(sel_dists / D, D / sel_dists)
            raw = np.mean(ratios)
            v_score = min(1.0, 1.0 / raw)
            if v_score < min_score:
                min_score = v_score
                if min_score == 0.0:
                    break
                    
        return min_score

    # Objective to minimize: -score
    def objective(radii):
        s = score_polygon(radii)
        return -s

    # Initial guess
    r0 = np.ones(n)
    
    # Optimization
    # Use Powell method which doesn't require gradients and handles box constraints
    # Bounds: radii > 0. Let's say 0.1 to 2.0
    bounds = [(0.1, 2.0)] * n
    
    best_score = score_polygon(r0)
    best_radii = r0
    
    # Run optimization
    # Set a time limit
    start_time = time.time()
    time_limit = 600.0
    
    # Powell can be slow. Let's try Nelder-Mead first with maxiter
    # Or just run Powell with a reasonable maxiter
    # We can also try multiple restarts if time permits
    
    # Let's try a few different initial guesses
    guesses = [
        np.ones(n),
        1.0 + 0.1 * np.cos(5 * angles),
        1.0 + 0.1 * np.cos(10 * angles),
        1.0 + 0.2 * np.sin(3 * angles)
    ]
    
    for g in guesses:
        if time.time() - start_time > time_limit - 30:
            break
        try:
            res = minimize(objective, g, method='Powell', bounds=bounds, options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            if -res.fun > best_score:
                best_score = -res.fun
                best_radii = res.x
        except Exception:
            pass
            
    # Final check and normalization
    x = best_radii * np.cos(angles)
    y = best_radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # The function returns the polygon. The evaluator normalizes it.
    # But we should ensure it's valid.
    return polygon

# EVOLVE-BLOCK-END
