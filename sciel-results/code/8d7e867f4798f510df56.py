# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _is_convex_turn(p, q, r):
    """Check if turn p->q->r is convex (left turn)."""
    return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]) > 1e-9

def _count_convex7(points):
    """Count the number of convex 7-gons in the point set."""
    n = len(points)
    # Sort points lexicographically
    indices = np.argsort(points[:, 0] * 10000 + points[:, 1])
    sorted_points = points[indices]
    
    total_count = 0
    
    # Precompute angles for each pivot? Or just sort in the loop.
    # n=33, so sorting 32 points 33 times is fine.
    
    for i in range(n):
        pivot = sorted_points[i]
        # Get points with index > i in the sorted order
        # These are the points "later" in lexicographic order
        later_indices = list(range(i + 1, n))
        if len(later_indices) < 6:
            continue
            
        later_points = sorted_points[later_indices]
        
        # Sort these points by angle around pivot
        angles = np.arctan2(later_points[:, 1] - pivot[1], later_points[:, 0] - pivot[0])
        angle_order = np.argsort(angles)
        
        # Map original later_indices to sorted order
        # sorted_later_indices[j] is the index in sorted_points of the j-th point in angle order
        sorted_later_indices = [later_indices[j] for j in angle_order]
        sorted_later_points = later_points[angle_order]
        
        m = len(sorted_later_indices)
        
        # DP[l][m_idx][j_idx] is too big?
        # We want chains of length 6 (7 points including pivot).
        # Chain: pivot -> p1 -> p2 -> p3 -> p4 -> p5 -> p6
        # Turns: (pivot, p1, p2), (p1, p2, p3), ..., (p5, p6, pivot), (p6, pivot, p1)
        
        # dp[l][a, b] where a is the index of the (l-1)-th point and b is the l-th point in the chain
        # l=1: pivot -> b. dp[1][i, b] = 1. (i is pivot index)
        # l=2: pivot -> a -> b. dp[2][a, b] = sum(dp[1][i, a] * is_convex(pivot, a, b))
        # ...
        
        # We only care about chains ending at some b, and we need to check closing turns.
        # Let's use a dictionary or sparse representation if needed, but n is small.
        
        # dp[l][a, b]
        # a, b are indices in sorted_later_indices (0 to m-1)
        
        # Initialize dp[1]
        # dp[1][i, b] is not really needed as a matrix, just that chain exists.
        
        # Let's define dp[l][a, b] as number of chains of length l ending with a->b.
        # l=1: pivot->b. a is implicit pivot.
        
        # For l=1, we have m chains: pivot -> sorted_later_points[b]
        # For l=2, a->b where a is from l=1.
        
        # We will store dp as a list of dictionaries or 2D arrays.
        # Given m <= 32, a 32x32 array is small.
        
        # dp[l] is a 2D array of size m x m
        # dp[l][a, b] = number of chains of length l ending with edge a->b
        
        # l=1:
        # The "edge" is pivot->b. We can represent this as dp[1][b] = 1?
        # No, we need the previous point for the turn.
        # Let's shift indices.
        # Chain of length l has l edges.
        # Edge 1: pivot -> p1
        # Edge 2: p1 -> p2
        # ...
        # Edge 6: p5 -> p6
        
        # dp[l][prev, curr]
        
        # l=1: dp[1][pivot_idx, curr] = 1. But pivot_idx is not in 0..m-1.
        # Let's just handle l=1 separately.
        
        # dp[2][a, b]: a is p1, b is p2.
        # dp[2][a, b] = 1 if turn(pivot, p_a, p_b) is convex.
        
        # dp[l][a, b]: a is p_{l-1}, b is p_l.
        # dp[l][a, b] = sum_{c} dp[l-1][c, a] * is_convex(p_c, p_a, p_b)
        
        # We need to check closing turns for l=6.
        # Chain: pivot -> p1 -> ... -> p6.
        # Turns to check:
        # 1. (pivot, p1, p2) - checked in dp[2]
        # 2. (p1, p2, p3) - checked in dp[3]
        # ...
        # 5. (p4, p5, p6) - checked in dp[6]
        # 6. (p5, p6, pivot)
        # 7. (p6, pivot, p1)
        
        # So for each valid chain ending at p6 (index b) coming from p5 (index a),
        # we check turn(p_a, p_b, pivot) and turn(p_b, pivot, p1).
        # But we don't know p1!
        
        # We must store p1.
        # dp[l][start, prev, curr]
        # l=1: start=pivot, prev=pivot, curr=p1.
        # This is getting large. m^3 * 6. 32^3 * 6 ~ 200k. Still fine.
        
        # Let's use dp[l][s, a, b] where s is the index of p1 (first point after pivot).
        # Actually, s is fixed for a given chain.
        
        # Alternative: Just iterate over all permutations of 6 points from the m later points?
        # P(32, 6) is huge.
        
        # Let's stick to DP with s.
        # dp[l][s, a, b]
        # l=1: dp[1][s, s, s] = 1? No.
        # Chain: pivot -> p_s.
        # Let's define state as (current_last, prev_last, first_point).
        
        # dp[l][first, prev, curr]
        # l=1: first=curr, prev=pivot.
        # We can't have pivot in the array indices.
        
        # Let's re-index.
        # Points are 0..m-1.
        # dp[1][first, curr]: first=curr. Value 1.
        # dp[2][first, prev, curr]: prev=first. Value = 1 if turn(pivot, p_first, p_curr) is convex.
        # dp[l][first, prev, curr]: Value = sum_{pp} dp[l-1][first, pp, prev] * is_convex(p_pp, p_prev, p_curr).
        
        # Size: m * m * m = 32^3 = 32768.
        # l goes up to 6.
        # Total operations: 6 * 32768 * 32 (sum over pp) ~ 6 million. Very fast.
        
        # Initialize dp[1]
        # dp1 is a 2D array m x m, dp1[first, curr] = 1 if first==curr else 0.
        
        # Actually, we can just use a dictionary for sparse states if needed, but dense is fine.
        
        # dp[l] will be a 3D array (first, prev, curr)
        
        # l=1:
        # We don't really need a 3D array for l=1.
        # Let's start l from 2.
        
        # Base: For each pair (a, b), if turn(pivot, p_a, p_b) is convex, then we have a chain pivot->a->b.
        # first=a, prev=a, curr=b.
        
        # dp2 = np.zeros((m, m, m), dtype=int)
        # for a in range(m):
        #   for b in range(m):
        #     if a != b and _is_convex_turn(pivot, sorted_later_points[a], sorted_later_points[b]):
        #       dp2[a, a, b] = 1
        
        # Then iterate l=3 to 6.
        
        # For l=6, we have dp6[first, prev, curr].
        # We need to check:
        # 1. turn(p_prev, p_curr, pivot) is convex.
        # 2. turn(p_curr, pivot, p_first) is convex.
        
        # If both are true, add to count.
        
        # Optimization: Precompute convexity matrix for all triples of (pivot, p_i, p_j) and (p_i, p_j, pivot) etc.
        
        # Let's implement this.
        
        # Precompute turn matrices for this pivot
        # T1[i, j] = 1 if turn(pivot, p_i, p_j) is convex
        # T2[i, j, k] = 1 if turn(p_i, p_j, p_k) is convex
        # T3[i, j] = 1 if turn(p_i, p_j, pivot) is convex
        # T4[i, j] = 1 if turn(p_j, pivot, p_i) is convex  (Note order: p_j -> pivot -> p_i)
        
        # T1 is m x m
        # T2 is m x m x m
        # T3 is m x m
        # T4 is m x m
        
        # m is at most 32.
        
        # T1:
        T1 = np.zeros((m, m), dtype=bool)
        for a in range(m):
            for b in range(m):
                if a != b:
                    T1[a, b] = _is_convex_turn(pivot, sorted_later_points[a], sorted_later_points[b])
        
        # T2:
        T2 = np.zeros((m, m, m), dtype=bool)
        for i in range(m):
            for j in range(m):
                if i == j: continue
                for k in range(m):
                    if j == k: continue
                    T2[i, j, k] = _is_convex_turn(sorted_later_points[i], sorted_later_points[j], sorted_later_points[k])
        
        # T3: turn(p_i, p_j, pivot)
        T3 = np.zeros((m, m), dtype=bool)
        for i in range(m):
            for j in range(m):
                if i != j:
                    T3[i, j] = _is_convex_turn(sorted_later_points[i], sorted_later_points[j], pivot)
        
        # T4: turn(p_j, pivot, p_i)
        T4 = np.zeros((m, m), dtype=bool)
        for i in range(m):
            for j in range(m):
                if i != j:
                    T4[i, j] = _is_convex_turn(sorted_later_points[j], pivot, sorted_later_points[i])
        
        # DP
        # dp[l][first, prev, curr]
        # l=2: dp2[first, first, curr] = T1[first, curr]
        
        dp = np.zeros((m, m, m), dtype=np.int64)
        for a in range(m):
            for b in range(m):
                if T1[a, b]:
                    dp[a, a, b] = 1
        
        # Iterate l=3 to 6
        for l in range(3, 7):
            new_dp = np.zeros((m, m, m), dtype=np.int64)
            # new_dp[first, prev, curr] = sum_{pp} dp[first, pp, prev] * T2[pp, prev, curr]
            # Vectorized sum over pp
            # dp is (first, pp, prev)
            # T2 is (pp, prev, curr)
            # We want to sum over pp.
            # new_dp[first, prev, curr] = sum_pp dp[first, pp, prev] * T2[pp, prev, curr]
            
            # This is a tensor contraction.
            # dp: (F, P, Q) where P=pp, Q=prev
            # T2: (P, Q, R) where R=curr
            # Result: (F, Q, R)
            
            # np.einsum('fpq, pqr -> fqr', dp, T2)
            
            new_dp = np.einsum('fpq,pqr->fqr', dp, T2, optimize=True)
            dp = new_dp
        
        # Now dp is for l=6. dp[first, prev, curr]
        # Check closing turns
        # T3[prev, curr] and T4[first, curr]
        
        # valid = dp * T3[prev, curr] * T4[first, curr]
        # Sum all.
        
        # T3 is (prev, curr)
        # T4 is (first, curr)
        
        # We need to broadcast T3 and T4 to match dp's shape (first, prev, curr).
        # T3[prev, curr] -> T3[np.newaxis, :, :] (1, prev, curr)
        # T4[first, curr] -> T4[:, np.newaxis, :] (first, 1, curr)
        
        valid_mask = T3[np.newaxis, :, :] * T4[:, np.newaxis, :]
        count = np.sum(dp * valid_mask)
        total_count += count
        
    return total_count

def _is_valid(points):
    """Check validity constraints."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check distances
    # n=33, 33*32/2 = 528 pairs.
    for i in range(33):
        for j in range(i + 1, 33):
            dist_sq = np.sum((points[i] - points[j]) ** 2)
            if dist_sq < 1e-12:
                return False
    # Check collinearity
    # 33 choose 3 = 5456 triples.
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                p1, p2, p3 = points[i], points[j], points[k]
                area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < 1e-9:
                    return False
    return True

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 600 # Safety margin
    
    # Initial Construction: Clustered near corners
    # 6 clusters of sizes 1, 2, 4, 8, 16, 12
    # Corners: (0,0), (1,0), (1,1), (0,1), (0.5, 0), (0.5, 1)
    
    np.random.seed(42)
    
    # Define cluster centers and sizes
    # We want to minimize convex 7-gons.
    # Placing points in convex chains near the boundary helps.
    
    # Let's try a specific known good construction or a heuristic one.
    # Heuristic: 2 large clusters of 16 and 17 points?
    # Or 6 clusters as above.
    
    # Let's generate points in 6 small convex chains near the 6 "sides/corners"
    
    points = np.zeros((n, 2))
    
    # Cluster 1: 1 point at (0.1, 0.1)
    points[0] = [0.1, 0.1]
    
    # Cluster 2: 2 points near (0.9, 0.1)
    points[1] = [0.8, 0.1]
    points[2] = [0.9, 0.2]
    
    # Cluster 3: 4 points near (0.9, 0.9)
    points[3] = [0.8, 0.8]
    points[4] = [0.9, 0.8]
    points[5] = [0.9, 0.9]
    points[6] = [0.8, 0.9]
    
    # Cluster 4: 8 points near (0.1, 0.9)
    # Make them a convex chain
    for i in range(8):
        # Arc from (0.1, 0.8) to (0.2, 0.9)
        t = i / 7.0
        points[7+i] = [0.1 + 0.1*t, 0.8 + 0.1*t]
        
    # Cluster 5: 16 points near (0.5, 0.0)
    for i in range(16):
        t = i / 15.0
        points[15+i] = [0.3 + 0.4*t, 0.05 + 0.05*np.sin(t*np.pi)]
        
    # Cluster 6: 12 points near (0.5, 1.0)
    for i in range(12):
        t = i / 11.0
        points[31-i] = [0.3 + 0.4*t, 0.95 - 0.05*np.sin(t*np.pi)] # 31-11=20 to 31
        
    # Ensure validity
    if not _is_valid(points):
        # Fallback to random if invalid
        points = np.random.rand(n, 2)
    
    best_points = points.copy()
    best_count = _count_convex7(points)
    
    # Hill Climbing
    # Time budget: 500s
    max_time = 500
    
    while time.time() - start_time < max_time:
        # Pick a random point to move
        idx = np.random.randint(n)
        old_point = points[idx].copy()
        
        # Move to a new random position
        # Or perturb? Random restart within square is more effective for escaping local minima.
        new_point = np.random.rand(2)
        points[idx] = new_point
        
        if not _is_valid(points):
            points[idx] = old_point
            continue
            
        new_count = _count_convex7(points)
        
        if new_count < best_count:
            best_count = new_count
            best_points = points.copy()
            if best_count == 1:
                break # Can't do better than 1
        else:
            # Revert
            points[idx] = old_point
            
    return best_points

# EVOLVE-BLOCK-END
