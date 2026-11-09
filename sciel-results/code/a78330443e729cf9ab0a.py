# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_vertex_score(vertex, all_vertices):
    """
    Compute the score for a single vertex based on the task contract.
    """
    # Distances to all other vertices
    diffs = all_vertices - vertex
    dists = np.linalg.norm(diffs, axis=1)
    
    # Exclude the vertex itself (distance 0)
    # The contract says "19 distances from v to the other vertices"
    # So we just take the 19 non-zero distances.
    # Since vertex is in all_vertices, one dist is 0. We sort and ignore the 0.
    dists_sorted = np.sort(dists)
    # dists_sorted[0] should be 0. We take the next 19.
    d = dists_sorted[1:20]
    
    # Check for very small distances
    if np.any(d < 0.05):
        return 0.0
        
    # Find the gap
    # k is the first index minimizing d_{k+1} - d_k
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    # Select the four distances
    if k == 0:
        selected = d[0:4]
    elif k == 17: # Since d has length 19, indices 0..18. Max k for diff is 17 (d[17]-d[18])
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    # Calculate raw score
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _calculate_global_score(vertices):
    """
    Calculate the minimum vertex score over all vertices.
    Also checks for convexity.
    """
    n = len(vertices)
    if n != 20:
        return 0.0
        
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(vertices[i] - vertices[j]) < 1e-9:
                return 0.0

    # Check convexity
    # Calculate cross products of consecutive edges
    # v0 -> v1 -> v2 ...
    # Edge e_i = v_{i+1} - v_i
    # Cross product of e_i and e_{i+1} must be all positive (or all negative)
    cross_products = []
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        p2 = vertices[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(cross) < 1e-9:
            return 0.0 # Not strictly convex (collinear)
        cross_products.append(cross)
        
    cross_products = np.array(cross_products)
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0 # Not convex

    # Calculate vertex scores
    min_score = 1.0
    for i in range(n):
        score = _compute_vertex_score(vertices[i], vertices)
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses radial modulation and coordinate descent to maximize the minimum vertex score.
    """
    n = 20
    # Initial radii (regular polygon)
    radii = np.ones(n)
    
    # Time limit guard
    start_time = time.time()
    time_limit = 500.0 # Leave margin for 600s limit
    
    # Coordinate Descent
    # We will iterate through each vertex and adjust its radius.
    # To be efficient, we can use a ternary search on the radius for each vertex.
    
    # Helper to get vertices from radii
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_vertices(r):
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    current_vertices = get_vertices(radii)
    current_score = _calculate_global_score(current_vertices)
    
    # If initial score is 0, try to fix it or just return regular polygon
    # Regular polygon should be valid.
    
    # Optimization Loop
    # We will perform multiple passes. In each pass, we update each radius.
    # For each radius update, we do a ternary search in a local window.
    
    num_passes = 200 # Adjust based on time
    window = 0.2 # Initial search window for radius
    
    for p in range(num_passes):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
                
            # Define function to evaluate score with radius i set to r
            def score_with_r(r_val):
                new_radii = radii.copy()
                new_radii[i] = r_val
                verts = get_vertices(new_radii)
                return _calculate_global_score(verts)
            
            # Ternary search for maximum score in [radii[i] - window, radii[i] + window]
            # Note: The score function is not necessarily unimodal, but locally it often is.
            # We can also try a grid search if ternary fails or is slow.
            # Given n=20, evaluating score is O(n^2 log n) ~ 400*11 = 4400 ops.
            # 20 vertices * 20 passes * 20 ternary steps = 8000 evaluations.
            # 8000 * 4400 ~ 35 million ops. This is very fast in Python/NumPy.
            
            lo = radii[i] - window
            hi = radii[i] + window
            
            # Ensure radii stay positive and reasonable to maintain convexity
            # A radius that is too small might cause non-convexity or self-intersection.
            # A radius that is too large is fine but might reduce score.
            # Let's clamp lo to 0.1 to avoid degenerate triangles.
            lo = max(lo, 0.1)
            
            # Ternary search
            # We want to maximize score.
            # f(x) is the score.
            # We find x that maximizes f(x).
            
            # Perform a coarse grid search first to find a good range?
            # Or just ternary.
            # Since the function is discrete (due to sorting and min), it might be noisy.
            # A grid search with ~50 points might be more robust and still fast.
            
            grid_size = 50
            grid_points = np.linspace(lo, hi, grid_size)
            best_r = radii[i]
            best_score = score_with_r(radii[i])
            
            for r_val in grid_points:
                s = score_with_r(r_val)
                if s > best_score:
                    best_score = s
                    best_r = r_val
                    
            if best_r != radii[i]:
                radii[i] = best_r
                improved = True
                
            # Optional: Refine with smaller window around best_r
            # window = window * 0.5 # Shrink window for next pass?
            
        if not improved:
            # Try shrinking window to find finer improvements
            window *= 0.8
            if window < 0.001:
                break
                
    # Final check and return
    final_vertices = get_vertices(radii)
    final_score = _calculate_global_score(final_vertices)
    
    # If the optimized polygon is invalid (score 0), fall back to regular polygon
    if final_score == 0.0:
        regular_radii = np.ones(n)
        final_vertices = get_vertices(regular_radii)
        
    return final_vertices

# EVOLVE-BLOCK-END
