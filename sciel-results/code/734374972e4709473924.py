import numpy as np
import time
from scipy.optimize import differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Start time for deadline guard
    start_time = time.time()
    deadline = 600.0  # 630s limit, keep 30s margin
    
    # Parameters: coefficients for cos(k*theta) and sin(k*theta) for k=1,2,3
    # Total 6 parameters
    # r(theta) = 1 + sum_{k=1}^3 (a_k cos(k theta) + b_k sin(k theta))
    
    # Bounds to ensure convexity (small perturbations)
    bounds = [(-0.05, 0.05)] * 6
    
    # Pre-compute angles
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    def generate_polygon(params):
        a1, b1, a2, b2, a3, b3 = params
        r = 1.0 + (a1 * np.cos(angles) + b1 * np.sin(angles)) + \
                    (a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)) + \
                    (a3 * np.cos(3 * angles) + b3 * np.sin(3 * angles))
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.column_stack((x, y))

    def is_convex(polygon):
        # Check if polygon is strictly convex
        # Cross product of consecutive edges must be all positive (or all negative)
        # and non-zero
        n = len(polygon)
        if n < 3:
            return False
        
        cross_prods = np.zeros(n)
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross_prods[i] = v1[0] * v2[1] - v1[1] * v2[0]
        
        # All must be same sign and non-zero
        if np.any(cross_prods == 0):
            return False
        if np.all(cross_prods > 0) or np.all(cross_prods < 0):
            return True
        return False

    def compute_vertex_scores(polygon):
        # Normalize polygon for scoring?
        # The prompt says the evaluator normalizes. 
        # But for the optimizer, we can compute on the raw polygon 
        # because the score is scale/translation invariant.
        # Let's verify invariance.
        # Distances scale with R. d_i -> R d_i.
        # D -> R D.
        # d_i/D is invariant.
        # So we can compute on raw coordinates.
        
        n = len(polygon)
        scores = np.zeros(n)
        
        for i in range(n):
            # Distances from vertex i to all others
            dists = np.linalg.norm(polygon - polygon[i], axis=1)
            # Exclude self (dist 0)
            dists = np.delete(dists, i)
            dists.sort()
            
            # d_0 ... d_18
            # Check if any < 0.05 (in normalized units? No, raw units?)
            # The prompt says: "If any is < 0.05 the vertex scores 0."
            # This threshold is likely in the *normalized* frame.
            # Since scale is arbitrary, I should normalize before checking thresholds.
            # Let's normalize the polygon first.
            
            # Actually, it's better to normalize the whole polygon once.
            # But doing it inside the loop is inefficient.
            # Let's pass normalized polygon to this function.
            
            # Find k minimizing d_{k+1} - d_k
            gaps = np.diff(dists)
            k = np.argmin(gaps)
            
            D = (dists[k] + dists[k+1]) / 2.0
            if D < 0.1: # Threshold in normalized units
                scores[i] = 0.0
                continue
            
            # Select 4 distances
            if k == 0:
                sel_dists = dists[0:4]
            elif k == 17:
                sel_dists = dists[15:19]
            else:
                sel_dists = dists[k-1:k+3]
                
            # raw = mean(max(d/D, D/d))
            ratios = np.maximum(sel_dists / D, D / sel_dists)
            raw = np.mean(ratios)
            
            if raw > 0:
                scores[i] = min(1.0, 1.0 / raw)
            else:
                scores[i] = 0.0
                
        return np.min(scores)

    def objective(params):
        # Check time
        if time.time() > deadline - 10:
            # Return a neutral value or stop?
            # DE handles this poorly if we just return. 
            # Better to have a global flag or just let it finish quickly.
            # For now, just return a bad score to discourage this path.
            pass
            
        polygon = generate_polygon(params)
        
        # Check convexity
        if not is_convex(polygon):
            return 1.0  # Maximize score, so minimize -score. Bad score = 0, so -0 = 0? No.
            # We want to maximize min_score.
            # objective = -min_score.
            # If invalid, min_score = 0. objective = 0.
            # But we want to penalize invalids more strongly to avoid them.
            # Let's return 1.0 (which is -(-1), implying score -1?)
            # Actually, DE minimizes.
            # Valid score range [0, 1].
            # Invalid score should be treated as 0.
            # So objective = -0 = 0.
            # But we might have valid polygons with score 0.01.
            # We want to avoid invalids.
            # Let's return a large positive number for invalids? No, DE minimizes.
            # So we want invalids to have a HIGH objective value.
            # If valid score is 0.5, obj = -0.5.
            # If invalid, obj = 10.0.
            return 10.0

        # Normalize polygon for scoring thresholds
        centroid = np.mean(polygon, axis=0)
        polygon_norm = polygon - centroid
        mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
        if mean_dist <= 0.01:
            return 10.0
        polygon_norm = polygon_norm / mean_dist
        
        score = compute_vertex_scores(polygon_norm)
        return -score

    # Run Differential Evolution
    # popsize=10, maxiter=100 might be too slow or too fast.
    # Let's try popsize=10, maxiter=50.
    # We can check time inside the callback? DE doesn't have a simple callback for stopping.
    # We can just run it and hope it finishes.
    # With 6 dimensions, it should be fast.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-5, 
            seed=42,
            polish=True
        )
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.column_stack((x, y))

    best_params = result.x
    best_polygon = generate_polygon(best_params)
    
    # Final check and normalization (though evaluator does it, let's return a nice one)
    # Actually, just return the coordinates.
    return best_polygon
