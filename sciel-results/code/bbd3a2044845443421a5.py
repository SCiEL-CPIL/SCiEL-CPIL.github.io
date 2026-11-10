# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This implementation uses a Fourier series perturbation of the regular polygon's radii,
    optimized via Differential Evolution to maximize the minimum vertex score.
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Pre-compute cos and sin for Fourier terms
    # We use frequencies 1 and 2. 
    # k=1: (a1, b1)
    # k=2: (a2, b2)
    # Total 4 parameters.
    
    # Bounds for Fourier coefficients. 
    # r_i = 1 + sum. We want r_i > 0.
    # If max perturbation is 0.5, r_i in [0.5, 1.5]. This is safe for convexity.
    bound = 0.5
    bounds = [(-bound, bound)] * 4
    
    def compute_polygon(params):
        a1, b1, a2, b2 = params
        # r_i = 1 + a1*cos(theta) + b1*sin(theta) + a2*cos(2*theta) + b2*sin(2*theta)
        r = 1.0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
        
        # Ensure radii are positive (though bounds should prevent negative)
        # If any r <= 0, return a dummy invalid polygon
        if np.any(r <= 0):
            return None
            
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        return polygon

    def is_valid_polygon(polygon):
        """Check strict convexity and normalization constraints."""
        if polygon is None:
            return False
        
        # 1. No repeated points
        # Check pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(polygon[i] - polygon[j]) < 1e-6:
                    return False
        
        # 2. Centroid and normalization
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return False
            
        # Normalize for convexity check? 
        # Convexity is scale invariant. We can check on centered points.
        
        # 3. Strict convexity: cross product of consecutive edges must be non-zero and same sign
        # Edges: e_i = p_{i+1} - p_i
        # Cross product in 2D: e_i x e_{i+1} = e_ix * e_{i+1,y} - e_iy * e_{i+1,x}
        # For CCW polygon, this should be positive.
        
        edges = np.roll(centered, -1, axis=0) - centered
        # Cross product of edge i and edge i+1
        # edge_i = edges[i]
        # edge_i+1 = edges[i+1]
        cross_products = np.zeros(n)
        for i in range(n):
            e1 = edges[i]
            e2 = edges[(i + 1) % n]
            cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
            
        # All cross products must be > 0 (or all < 0). 
        # Since we start from CCW regular polygon, they should be positive.
        if np.any(cross_products <= 1e-9): # Small threshold for "nonzero"
            return False
            
        return True

    def vertex_score(polygon, v_idx):
        """Calculate the score for a single vertex."""
        v = polygon[v_idx]
        # Distances to all other vertices
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self distance (which is 0)
        # The task says "19 distances from v to the other vertices"
        # So we remove the 0th element (distance to self)
        # Note: polygon[v_idx] is at index v_idx. 
        # dists[v_idx] is 0.
        other_dists = np.delete(dists, v_idx)
        
        # Sort distances
        sorted_dists = np.sort(other_dists)
        
        # Check for any < 0.05
        if sorted_dists[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        # d_0 <= ... <= d_18
        # Gaps: d_1-d_0, ..., d_18-d_17
        # We need the gap that is the "largest"? 
        # "Let k be the first index minimizing d_{k+1} - d_k"
        # Wait, "minimizing d_{k+1} - d_k" means the SMALLEST gap?
        # Let's re-read carefully: "Let k be the first index minimizing d_{k+1} - d_k"
        # Usually, we look for the largest gap to separate clusters. 
        # If we minimize the gap, we find the tightest pair.
        # Then D = (d_k + d_{k+1})/2.
        # If D < 0.1, score 0.
        # Then take 4 distances around this gap.
        
        # Let's check the regular polygon case.
        # Distances from a vertex in regular 20-gon:
        # Chord lengths: 2R sin(m * pi / 20) for m=1..19.
        # m=1,2,3,4 are the smallest.
        # Gaps between consecutive sorted distances:
        # The gaps are roughly constant for small m, then grow.
        # The smallest gap is likely between m=1 and m=2? Or m=9 and m=10?
        # Actually, the gaps increase as m increases until the middle.
        # So the smallest gap is at the beginning?
        # Let's calculate for regular polygon.
        # If k minimizes the gap, it picks the tightest cluster.
        # For regular polygon, the "cluster" of 4 nearest neighbors is m=1,2,3,4.
        # The gaps are d_1-d_0, d_2-d_1, d_3-d_2, d_4-d_3.
        # Are these the smallest gaps?
        # Yes, chord length growth is slow at first.
        # So k will likely be 0, 1, 2, or 3.
        
        gaps = np.diff(sorted_dists)
        # Find index of minimum gap
        k = np.argmin(gaps)
        
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        # Take four distances:
        # d_0..d_3 if k = 0
        # d_15..d_18 if k = 17 (max index for gap is 17, since 19 dists -> 18 gaps? No, 19 dists -> 18 gaps. Indices 0..17. Max k=17)
        # else d_{k-1}..d_{k+2}
        
        if k == 0:
            selected_dists = sorted_dists[0:4]
        elif k == 17:
            selected_dists = sorted_dists[15:19]
        else:
            # k is in 1..16
            # d_{k-1}, d_k, d_{k+1}, d_{k+2}
            selected_dists = sorted_dists[k-1:k+3]
            
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        
        if raw < 1e-9:
            return 0.0
            
        score = min(1.0, 1.0 / raw)
        return score

    def objective(params):
        """
        Objective function for optimization.
        Returns negative of the minimum vertex score.
        Penalizes invalid polygons heavily.
        """
        polygon = compute_polygon(params)
        if polygon is None:
            return 1000.0
            
        if not is_valid_polygon(polygon):
            return 1000.0
            
        # Calculate scores for all vertices
        scores = []
        for i in range(n):
            s = vertex_score(polygon, i)
            scores.append(s)
            
        min_score = np.min(scores)
        
        # We want to maximize min_score.
        # Return negative for minimization.
        return -min_score

    # Optimization
    # Deadline guard
    start_time = time.time()
    timeout = 600.0
    margin = 30.0
    
    # Use Differential Evolution
    # maxiter and popsize control the search.
    # We have 4 parameters. 
    # Let's use a reasonable population size.
    # If time is tight, DE might not finish. 
    # We can set a callback to check time.
    
    def time_check(params, convergence):
        elapsed = time.time() - start_time
        if elapsed > timeout - margin:
            raise Exception("Timeout")

    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            callback=time_check
        )
    except Exception:
        # If timeout or other error, fall back to best found or regular polygon
        # DE doesn't easily return best on exception in all versions, 
        # but we can just use the regular polygon if it fails.
        # Actually, we can store best result manually if needed, but DE is simple.
        # For safety, let's just return the regular polygon if it fails.
        # Or better, we can run a simpler local search if DE fails?
        # No, let's stick to DE. If it times out, it's better to have a valid polygon than crash.
        # But we want to beat 1.0.
        # Let's try to catch the timeout and use the last known good?
        # Differential Evolution in SciPy doesn't expose intermediate bests easily in callback.
        # However, if it raises, we don't have the result.
        
        # Alternative: Use a fixed number of iterations that is fast.
        # 4 params is small. 100 iterations of DE with popsize 15 is 1500 evaluations.
        # Each eval is O(N^2) ~ 400 ops. Very fast.
        # It should finish well within 600s.
        pass

    best_params = result.x
    best_polygon = compute_polygon(best_params)
    
    # Final validation and normalization
    if best_polygon is None or not is_valid_polygon(best_polygon):
        # Fallback to regular polygon
        r = np.ones(n)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
    else:
        polygon = best_polygon
        
    # Normalize as per task: subtract centroid, divide by mean distance to it
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        # Fallback
        r = np.ones(n)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
    normalized_polygon = centered / mean_dist
    
    return normalized_polygon


# EVOLVE-BLOCK-END
