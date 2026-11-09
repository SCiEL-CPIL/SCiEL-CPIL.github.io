import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    n = 20
    # Base angles and radii for regular polygon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_radii = np.ones(n)
    
    # Fourier modes to use
    K = 4
    # Parameters: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
    # a, b for angles; c, d for radii
    n_params = 4 * K
    
    def params_to_polygon(params):
        angles = base_angles.copy()
        radii = base_radii.copy()
        for k in range(1, K + 1):
            a = params[4*(k-1)]
            b = params[4*(k-1)+1]
            c = params[4*(k-1)+2]
            d = params[4*(k-1)+3]
            fk = k * base_angles
            angles += a * np.cos(fk) + b * np.sin(fk)
            radii += c * np.cos(fk) + d * np.sin(fk)
        
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    def vertex_score(polygon, i):
        v = polygon[i]
        # Distances to all other vertices
        diffs = polygon - v
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf # Exclude self
        
        # Sort distances
        sorted_dists = np.sort(dists)
        
        # Check for very small distances (degeneracy)
        if sorted_dists[0] < 0.05:
            return 0.0
            
        # Find gap
        # d_0 <= ... <= d_18
        # We need to find k such that d_{k+1} - d_k is minimized?
        # "Let k be the first index minimizing d_{k+1} - d_k"
        # Wait, the text says: "Let k be the first index minimizing d_{k+1} - d_k"
        # This implies we look for the smallest gap.
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1}) / 2
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        # Take four distances:
        # d_0..d_3 if k = 0
        # d_15..d_18 if k = 17
        # else d_{k-1}..d_{k+2}
        
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17: # Note: there are 19 distances, indices 0..18. 
                       # Gaps are between 0-1, ..., 17-18. So k is 0..17.
                       # If k=17, gap is between d_17 and d_18.
                       # The text says "d_15..d_18 if k = 17".
                       # Let's check the range. 19 distances.
                       # If k=17, we take d_15, d_16, d_17, d_18.
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        d_vals = sorted_dists[idxs]
        
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(d_vals / D, D / d_vals)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)

    def compute_score(polygon):
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
            
        normalized = centered / mean_dist
        
        # Check convexity
        # Cross products of consecutive edges
        edges = np.roll(normalized, -1, axis=0) - normalized
        # Edge i is p_{i+1} - p_i. 
        # Cross product of edge i and edge i+1
        cross_prods = np.cross(edges, np.roll(edges, -1, axis=0))
        
        # Strictly convex: all cross prods > 0 (for CCW)
        # If any is <= 0, it's not strictly convex in the CCW sense.
        # We assume the ordering is preserved.
        if np.any(cross_prods <= 0):
            # Penalty for non-convexity
            # We want to maximize score, so we return a negative penalty
            penalty = np.sum(np.maximum(0, -cross_prods)**2)
            # Also check for very small cross products (flat angles)
            flat_penalty = np.sum(np.maximum(0, 1e-5 - np.abs(cross_prods))**2)
            return - (penalty + flat_penalty)
            
        scores = []
        for i in range(n):
            s = vertex_score(normalized, i)
            scores.append(s)
            
        return min(scores)

    def objective(params):
        polygon = params_to_polygon(params)
        score = compute_score(polygon)
        return -score # Minimize negative score

    # Multi-start optimization
    best_score = 0.0
    best_params = np.zeros(n_params)
    
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline - 60: # Leave 60s for final check
            break
            
        if i == 0:
            params = np.zeros(n_params)
        else:
            # Random small perturbations
            params = np.random.normal(0, 0.1, n_params)
            
        # Use L-BFGS-B
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_params = res.x
        except Exception:
            pass
            
    # Final polygon
    polygon = params_to_polygon(best_params)
    
    # One last check: ensure it's valid and return it
    # If best_score is very low, maybe fallback to regular polygon?
    # But regular polygon score is 1.0.
    # If optimization failed, best_score might be 0 or negative.
    # We should return the regular polygon if the optimized one is worse.
    
    reg_polygon = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
    reg_score = compute_score(reg_polygon)
    
    if best_score < reg_score:
        return reg_polygon
    else:
        return polygon

