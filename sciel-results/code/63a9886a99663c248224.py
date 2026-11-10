import numpy as np
import time
import itertools

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    n = 33
    
    # Helper: Check validity strictly
    def is_valid(points):
        if points.shape != (n, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        
        # Distance check
        # Vectorized pairwise squared distances
        # d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dist_sq = np.sum(d**2, axis=-1)
        # if np.min(dist_sq[np.triu_indices(n, k=1)]) < 1e-12: return False
        
        # For speed in the inner loop, we might skip this or do it less frequently,
        # but the evaluator is strict. Let's do a fast check.
        # 33x33 matrix is small.
        d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=-1)
        # Exclude diagonal
        dist_sq[np.diag_indices(n)] = 1e18
        if np.min(dist_sq) < 1e-12:
            return False
            
        # Collinearity check
        # Check all triples for small area
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # We need |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| >= 1e-9
        # Vectorized:
        # i, j, k are indices
        # We can iterate or vectorize. 33 choose 3 is ~5500. 
        # Vectorizing all triples at once:
        # indices = np.array(list(itertools.combinations(range(n), 3)))
        # p1 = points[indices[:, 0]]
        # p2 = points[indices[:, 1]]
        # p3 = points[indices[:, 2]]
        # areas = np.abs(p1[:,0]*(p2[:,1]-p3[:,1]) + p2[:,0]*(p3[:,1]-p1[:,1]) + p3[:,0]*(p1[:,1]-p2[:,1]))
        # if np.any(areas < 1e-9): return False
        
        # To save time in the main loop, we will only do the full collinearity check
        # when a move is accepted, or periodically. 
        # However, if we accept a move that creates collinearity, the final result is invalid.
        # So we must ensure the final result is valid.
        # During search, we can assume small perturbations won't create exact collinearity
        # if we start from a non-degenerate state, but it's risky.
        # Let's do the collinearity check in the validation step of the search.
        
        return True

    def fast_is_valid(points):
        # Used for quick rejection in inner loops
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distance check is O(N^2) vectorized, very fast
        d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=-1)
        dist_sq[np.diag_indices(n)] = 1e18
        if np.min(dist_sq) < 1e-12:
            return False
        return True

    def count_convex_7goms(points):
        """
        Counts the number of 7-point subsets in convex position.
        Uses the definition: A subset is in convex position if no point lies inside
        the convex hull of the others.
        
        For a set of 7 points, it is NOT convex if at least one point is inside the
        convex hull of the other 6.
        
        Checking if a point is inside the convex hull of 6 others is expensive.
        Alternative: 
        A set of points is in convex position iff the number of points on the convex
        hull is equal to the size of the set.
        
        Efficient check for a specific subset of 7 points:
        1. Compute convex hull of the 7 points.
        2. If hull size == 7, it's convex.
        
        Doing this for C(33, 7) = 4.3M subsets is too slow in pure Python loops.
        We need a vectorized approach or a smarter counting method.
        
        Insight from inspirations: 
        "Dynamic program over convex chains" is mentioned in the task description for C.
        But implementing that from scratch is complex.
        
        Let's use a proxy that is closer to the truth but faster.
        Actually, let's try to vectorize the "point in convex hull" check for all subsets?
        No, memory is too high.
        
        Let's use the fact that if we minimize convex 4-gons, we likely minimize convex 7-gons.
        But the current program already does that and scores 2.
        The high scoring programs (136) likely use a better initial structure.
        
        Let's look at the "grid" idea.
        If we place points on a grid, many 7-subsets will contain interior points.
        
        Let's implement a faster counter for convex 7-gons using a sampling or approximation?
        No, the score depends on the EXACT count.
        
        We need an exact, fast counter.
        
        Let's use the property:
        C = sum_{S in C(33,7)} [S is convex]
        
        We can iterate over all 7-subsets in a vectorized manner if we chunk them.
        4.3M subsets * 7 points * 2 coords = 60M floats. ~500MB. Too much for one array.
        But we can process in batches of, say, 100,000 subsets.
        
        For each batch of subsets (shape B x 7 x 2):
        1. For each subset, check if all 7 points are on the hull.
        2. How to check hull size vectorized?
           For a set of points P (B x 7 x 2), a point P[i, j] is on the hull if it is not
           inside the convex hull of the other 6 points.
           
           This is still hard to vectorize fully without loops over the 7 points.
           However, 7 is small.
           For each subset, we can check:
           Is P0 inside hull(P1..P6)?
           Is P1 inside hull(P0, P2..P6)?
           ...
           
           Checking if a point is inside a convex polygon of 6 vertices can be done by
           checking if it is inside any of the triangles formed by the centroid and edges?
           Or simply: A point is inside the convex hull of 6 points iff it is inside
           the triangle formed by any 3 of them? No.
           
           Actually, for a set of 7 points, it is NON-convex iff there exists a point
           that is strictly inside the convex hull of the other 6.
           
           Let's use a simpler check:
           A 7-gon is convex iff its area is equal to the sum of the areas of the
           triangles formed by a fixed interior point (if it existed) and the edges?
           No.
           
           Let's go back to the "convex 4-gon" proxy but improve the search.
           The current program uses random initialization.
           Let's use a **Grid Initialization** and **Grid-based Local Search**.
           
           Why?
           Points on a grid have many "interior" relationships.
           A random cluster has points that are more "uniformly distributed" in the local neighborhood,
           leading to more convex hulls.
           
           Let's stick to the convex 4-gon proxy for speed, but use a much better initialization.
           And we will verify with a slower, exact convex 7-gon counter at the end of the search
           or periodically to ensure we are on the right track.
           
           Wait, the score is based on the EXACT convex 7-gon count.
           If I optimize 4-gons, I might not minimize 7-gons perfectly.
           But 4-gons are a strong lower bound indicator.
           
           Let's try to implement the exact 7-gon counter in batches.
           """
        count = 0
        # Generate indices
        # To avoid generating 4.3M indices in memory at once, we process in chunks.
        # But itertools.combinations is slow.
        # Let's pre-generate all combinations once? 
        # 4.3M * 7 * 4 bytes (int32) = 120MB. Acceptable.
        
        if not hasattr(count_convex_7goms, 'all_combos'):
            count_convex_7goms.all_combos = np.array(list(itertools.combinations(range(n), 7)), dtype=np.int32)
            
        combos = count_convex_7goms.all_combos
        num_combos = len(combos)
        
        # Process in batches of 50,000
        batch_size = 50000
        for start in range(0, num_combos, batch_size):
            end = min(start + batch_size, num_combos)
            batch_idx = combos[start:end]
            # Shape: (B, 7, 2)
            pts_batch = points[batch_idx]
            
            # For each of the 7 points in the subset, check if it is inside the convex hull of the other 6.
            # If ANY point is inside, the subset is NOT convex.
            # So we want to count subsets where NO point is inside.
            
            # Vectorized check:
            # For each point j in the subset (0..6), check if pts_batch[:, j] is inside hull(pts_batch[:, [0..6]\{j}])
            
            # Function: is_point_in_convex_hull(pt, hull_pts)
            # pt: (B, 2)
            # hull_pts: (B, 6, 2)
            
            def is_inside_hull(pt, hull_pts):
                # hull_pts is convex? No, the 6 points might not be in convex position themselves.
                # But the convex hull of the 6 points is a polygon with <= 6 vertices.
                # A point is inside the convex hull of a set of points iff it is inside
                # the convex hull of their convex hull vertices.
                # We can check if the point is inside ANY triangle formed by 3 of the hull points?
                # No, that's not sufficient.
                
                # Robust method: 
                # A point P is inside the convex hull of points Q0..Q5 iff
                # P can be written as a convex combination of Q0..Q5.
                # This is equivalent to: For every direction vector v, 
                # dot(P, v) <= max(dot(Qi, v)) for all i.
                # This is an infinite number of checks.
                
                # Alternative: 
                # Since 6 points are few, we can check if P is inside the triangle of any 3 of them?
                # No.
                
                # Let's use the property: 
                # P is inside hull(Q) iff the area of hull(Q) is equal to the sum of areas
                # of triangles (P, Qi, Qi+1) for all edges (Qi, Qi+1) of the hull.
                # But we don't know the hull vertices.
                
                # Simpler: 
                # If P is inside the convex hull of Q, then P is inside the triangle formed by 
                # any 3 points of Q that form a triangle containing P?
                # Actually, a set of points is in convex position iff no point is inside the 
                # convex hull of the others.
                
                # Let's use a different approach for the batch:
                # For a subset of 7 points, it is convex iff the number of points on the 
                # convex hull is 7.
                # We can compute the convex hull of the 7 points for each batch member.
                # This is hard to vectorize.
                
                # Let's fall back to the 4-gon proxy for the main search loop.
                # It is fast and correlates well.
                pass
            
            # Since exact 7-gon counting is too slow for the inner loop, 
            # we will use the 4-gon proxy for the search, 
            # but we will use a BETTER initialization.
            
            # For now, this function is a placeholder. 
            # We will implement a separate fast proxy.
            pass
            
        return 0 # Placeholder

    def count_convex_4goms(points):
        """
        Fast vectorized count of convex 4-gons.
        """
        if not hasattr(count_convex_4goms, 'all_combos'):
            count_convex_4goms.all_combos = np.array(list(itertools.combinations(range(n), 4)), dtype=np.int32)
            
        combos = count_convex_4goms.all_combos
        pts_4 = points[combos]
        
        p0 = pts_4[:, 0, :]
        p1 = pts_4[:, 1, :]
        p2 = pts_4[:, 2, :]
        p3 = pts_4[:, 3, :]
        
        def inside_tri(pt, a, b, c):
            d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
            d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
            d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
            return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
        
        in_0 = inside_tri(p0, p1, p2, p3)
        in_1 = inside_tri(p1, p0, p2, p3)
        in_2 = inside_tri(p2, p0, p1, p3)
        in_3 = inside_tri(p3, p0, p1, p2)
        
        is_convex = ~(in_0 | in_1 | in_2 | in_3)
        return np.sum(is_convex)

    # Initialization: Dense Grid
    # 33 points. 6x6 grid has 36 points. We can take a 6x6 grid and remove 3 points,
    # or use a 5x7 grid (35 points) and remove 2.
    # Let's use a 6x6 grid scaled to [0.1, 0.9] and remove 3 random interior points?
    # Or just use the first 33 points of a 6x6 grid.
    
    # Grid points
    x_coords = np.linspace(0.1, 0.9, 6)
    y_coords = np.linspace(0.1, 0.9, 6)
    grid_points = np.array([[x, y] for y in y_coords for x in x_coords])
    
    # Take first 33
    pts = grid_points[:33].copy()
    
    # Add small random jitter to break symmetries and ensure general position (no collinearity)
    np.random.seed(42)
    jitter = np.random.normal(0, 0.005, (n, 2))
    pts += jitter
    pts = np.clip(pts, 0.0, 1.0)
    
    # Ensure validity
    if not is_valid(pts):
        # Fallback to random cluster if grid fails validity (unlikely with jitter)
        pts = np.random.rand(n, 2) * 0.2 + 0.4
        pts += np.random.normal(0, 0.001, (n, 2))
        pts = np.clip(pts, 0.0, 1.0)
    
    current_score = count_convex_4goms(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Local Search
    # Move one point at a time.
    # Use delta calculation for 4-gons containing the moved point.
    
    def delta_convex_4goms(points, idx, new_p):
        others = [i for i in range(n) if i != idx]
        if not hasattr(delta_convex_4goms, 'combos_cache'):
            # Pre-compute combinations of 3 others for each idx? 
            # There are 33 idxs. Each has C(32,3) = 4960 combos.
            # Total 33 * 4960 = 163k combos. 
            # We can generate on the fly or cache.
            delta_convex_4goms.combos_cache = {}
            
        if idx not in delta_convex_4goms.combos_cache:
            delta_convex_4goms.combos_cache[idx] = np.array(list(itertools.combinations(others, 3)), dtype=np.int32)
            
        combos = delta_convex_4goms.combos_cache[idx]
        
        # Get the 3 other points for each combo
        p1 = points[combos[:, 0]]
        p2 = points[combos[:, 1]]
        p3 = points[combos[:, 2]]
        
        old_p = points[idx]
        old_p_exp = np.tile(old_p, (len(combos), 1))
        new_p_exp = np.tile(new_p, (len(combos), 1))
        
        def count_convex_in_batch(p_moving, p1, p2, p3):
            def inside_tri(pt, a, b, c):
                d1 = (b[:, 0] - a[:, 0]) * (pt[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (pt[:, 0] - a[:, 0])
                d2 = (c[:, 0] - b[:, 0]) * (pt[:, 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (pt[:, 0] - b[:, 0])
                d3 = (a[:, 0] - c[:, 0]) * (pt[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (pt[:, 0] - c[:, 0])
                return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
            
            in_0 = inside_tri(p_moving, p1, p2, p3)
            in_1 = inside_tri(p1, p_moving, p2, p3)
            in_2 = inside_tri(p2, p_moving, p1, p3)
            in_3 = inside_tri(p3, p_moving, p1, p2)
            
            is_convex = ~(in_0 | in_1 | in_2 | in_3)
            return np.sum(is_convex)
        
        old_count = count_convex_in_batch(old_p_exp, p1, p2, p3)
        new_count = count_convex_in_batch(new_p_exp, p1, p2, p3)
        
        return new_count - old_count

    no_improve = 0
    max_no_improve = 100
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    while time.time() < deadline:
        improved = False
        order = np.random.permutation(n)
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            
            old_p = best_pts[idx].copy()
            best_delta = 0
            best_new_p = None
            
            for step in step_sizes:
                for _ in range(8): # 8 random directions
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    new_p = old_p + np.array([dx, dy])
                    new_p = np.clip(new_p, 0.0, 1.0)
                    
                    # Quick validity check
                    test_pts = best_pts.copy()
                    test_pts[idx] = new_p
                    if not fast_is_valid(test_pts):
                        continue
                    
                    delta = delta_convex_4goms(best_pts, idx, new_p)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_p = new_p
                        improved = True
            
            if best_new_p is not None:
                best_pts[idx] = best_new_p
                best_score += best_delta
                no_improve = 0
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            # Perturb a random point significantly to escape local minima
            idx = np.random.randint(n)
            new_p = best_pts[idx] + np.random.normal(0, 0.05, 2)
            new_p = np.clip(new_p, 0.0, 1.0)
            test_pts = best_pts.copy()
            test_pts[idx] = new_p
            if fast_is_valid(test_pts):
                best_pts[idx] = new_p
                best_score = count_convex_4goms(best_pts)
            no_improve = 0

    # Final Validation
    def check_collinearity(points):
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    if not check_collinearity(best_pts):
        for _ in range(100):
            idx = np.random.randint(n)
            best_pts[idx] += np.random.normal(0, 0.001, 2)
            best_pts[idx] = np.clip(best_pts[idx], 0.0, 1.0)
            if fast_is_valid(best_pts) and check_collinearity(best_pts):
                break
    
    if not fast_is_valid(best_pts) or not check_collinearity(best_pts):
        pts = np.random.rand(n, 2) * 0.9 + 0.05
        return pts

    return best_pts
