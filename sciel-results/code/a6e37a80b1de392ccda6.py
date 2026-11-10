# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.

    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    k = 7
    deadline = time.time() + 600  # 600 second budget

    def is_valid(points):
        """Check if points are valid: correct shape, no NaN/inf, in range, no close points, no collinear."""
        if points.shape != (n, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Check no two points too close
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx * dx + dy * dy < 1e-12:
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for l in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[l]
                    area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    if abs(area) < 1e-9:
                        return False
        return True

    def count_convex7(points):
        """Count convex 7-gons using dynamic programming over convex chains."""
        # Sort lexicographically
        idx = np.lexsort((points[:, 1], points[:, 0]))
        sorted_points = points[idx]
        
        C = 0
        # For each pivot point
        for pivot_idx in range(n):
            pivot = sorted_points[pivot_idx]
            # Get all other points
            others = np.delete(sorted_points, pivot_idx, axis=0)
            # Sort by atan2 angle around pivot
            angles = np.arctan2(others[:, 1] - pivot[1], others[:, 0] - pivot[0])
            order = np.argsort(angles)
            sorted_others = others[order]
            
            # Now we have m = n-1 = 32 points sorted by angle around pivot
            # We want to count convex chains of length 7 that start and end at the pivot
            # This is equivalent to counting subsets of 6 points from sorted_others
            # that form a convex chain when connected to the pivot
            
            m = len(sorted_others)  # 32
            
            # DP: dp[i][j] = number of convex chains of length j ending at point i
            # where point i is in sorted_others
            # A chain p0 -> p1 -> ... -> pj is convex if each turn is left-turn
            # The full convex 7-gon is: pivot -> p0 -> p1 -> ... -> p5 -> pivot
            # We need:
            # 1. pivot -> p0 -> p1 is convex (left turn at p0)
            # 2. p0 -> p1 -> p2 is convex (left turn at p1)
            # ...
            # 6. p4 -> p5 -> pivot is convex (left turn at p5)
            # 7. p5 -> pivot -> p0 is convex (left turn at pivot)
            
            # Actually, let me think more carefully. A convex 7-gon has vertices
            # v0, v1, ..., v6 in order. If we fix v0 = pivot, then v1, ..., v6 are
            # chosen from the other points and sorted by angle.
            
            # For a subset {v1, ..., v6} to form a convex 7-gon with pivot:
            # - They must be in angular order around pivot
            # - The polygon v0->v1->...->v6->v0 must be convex
            # - This means all turns must be left turns (for CCW orientation)
            
            # Let's use DP where we track chains of increasing length
            # dp[i][len] = number of ways to form a convex chain of 'len' vertices
            # ending at sorted_others[i], where the chain starts from some earlier point
            
            # Actually, let me use a different approach:
            # For each subset of 6 points from the 32, check if they form a convex 7-gon with pivot
            # But that's C(32,6) = 906192 per pivot, times 33 pivots = 29.9M checks
            # Each check is O(7) = 49 operations, so ~1.5B operations. Might be slow.
            
            # Better: DP over subsets in angular order
            # dp[i][j] = number of convex chains of length j ending at point i
            # A chain is valid if all consecutive triples make left turns
            
            # Let's define:
            # dp[i][1] = 1 for all i (a chain of length 1 is just the point itself)
            # dp[i][j] = sum over h < i of dp[h][j-1] if the turn at h from the previous
            #   point to h to i is convex (left turn)
            
            # But we need to know the previous point too. So:
            # dp[i][j][h] would be too large.
            
            # Alternative: since points are sorted by angle, a convex chain must use
            # points in increasing index order. So we can do:
            # dp[i][j] = number of convex chains of length j ending at index i
            # dp[i][1] = 1
            # dp[i][j] = sum_{h < i} dp[h][j-1] * I(turn at h is convex)
            # where I(turn at h is convex) checks if the last two points in the chain
            # plus point i make a left turn. But we don't track the last two points.
            
            # We need: dp[i][j][last] = number of chains of length j ending at i with
            # previous point being last. This is O(m^2 * k) = 32^2 * 6 = 6144 states.
            
            # Let's do it properly:
            # dp[i][j] = dict mapping last_point_index -> count
            # But that's complex. Let me use a 2D array:
            # dp[i][j] = number of convex chains of length j ending at point i
            # To compute dp[i][j], we need to know which h < i can precede i in a
            # convex chain of length j-1. The condition is that for the chain ending
            # at h, the turn from the point before h to h to i is convex.
            
            # This requires knowing the point before h. So let's use:
            # dp[i][j] = sum over h < i of (number of convex chains of length j-1 ending at h
            #   with the property that the turn from prev_to_h to h to i is convex)
            
            # Since we don't track prev_to_h, let's use a different DP:
            # Let f(i, j) = number of convex chains of length j that end at point i
            # and whose second-to-last point is some h < i.
            # We can compute this by iterating.
            
            # Actually, the standard approach for counting convex k-gons in a point set
            # sorted by angle is:
            # For each starting point s, count the number of convex chains of length k-1
            # starting from s, where each consecutive triple makes a left turn.
            
            # Let me just do it with a simple DP that tracks the last two points:
            # dp[i][j] = number of convex chains of length j ending at point i
            # dp[i][1] = 1
            # dp[i][2] = i (number of points before i)
            # dp[i][j] = sum_{h < i} dp[h][j-1] * I(h, i, next) -- no, this doesn't work.
            
            # Let me use a different formulation. For counting convex k-gons where
            # one vertex is fixed as the pivot, and the other k-1 vertices are chosen
            # from points sorted by angle around the pivot:
            
            # A convex k-gon with vertices v0=pivot, v1, ..., v_{k-1} in angular order
            # is valid if and only if:
            # 1. The points v1, ..., v_{k-1} are in increasing angular order (guaranteed by sorting)
            # 2. For each consecutive triple (v_{i-1}, v_i, v_{i+1}) with indices mod k,
            #    the turn is convex (left turn for CCW polygon)
            
            # Since v0 is the pivot, the conditions are:
            # - Turn at v1: (v0, v1, v2) is left turn
            # - Turn at v2: (v1, v2, v3) is left turn
            # - ...
            # - Turn at v_{k-2}: (v_{k-3}, v_{k-2}, v_{k-1}) is left turn
            # - Turn at v_{k-1}: (v_{k-2}, v_{k-1}, v0) is left turn
            # - Turn at v0: (v_{k-1}, v0, v1) is left turn
            
            # So we need to count subsets {i1 < i2 < ... < i_{k-1}} from {0, ..., m-1}
            # such that all these turn conditions hold.
            
            # This can be done with DP. Let's define:
            # dp[i1][i2][...][ij] = 1 if the chain (pivot, s[i1], s[i2], ..., s[ij])
            # is a valid convex chain so far (all turns up to the last are convex)
            
            # But this is O(m^k) which is too large.
            
            # Instead, let's use the fact that for a fixed pivot, we can do:
            # dp[i][j] = number of convex chains of length j ending at index i
            # where the chain starts from some point and all internal turns are convex.
            # We add the pivot at the beginning and end.
            
            # Let me just brute force with optimization. For each pivot, we have 32 points.
            # We need to choose 6 of them. C(32,6) = 906192. For each, we check 7 turns.
            # Total: 33 * 906192 * 7 ≈ 209M operations. With NumPy, this might be feasible
            # if we vectorize.
            
            # Let's try a vectorized approach for each pivot.
            
            m = len(sorted_others)  # 32
            
            # Generate all combinations of 6 indices
            from itertools import combinations
            for combo in combinations(range(m), 6):
                i1, i2, i3, i4, i5, i6 = combo
                v1 = sorted_others[i1]
                v2 = sorted_others[i2]
                v3 = sorted_others[i3]
                v4 = sorted_others[i4]
                v5 = sorted_others[i5]
                v6 = sorted_others[i6]
                
                # Check all 7 turns
                # Turn at v1: (pivot, v1, v2)
                cross1 = (v1[0] - pivot[0]) * (v2[1] - v1[1]) - (v1[1] - pivot[1]) * (v2[0] - v1[0])
                if cross1 <= 1e-9:
                    continue
                
                # Turn at v2: (v1, v2, v3)
                cross2 = (v2[0] - v1[0]) * (v3[1] - v2[1]) - (v2[1] - v1[1]) * (v3[0] - v2[0])
                if cross2 <= 1e-9:
                    continue
                
                # Turn at v3: (v2, v3, v4)
                cross3 = (v3[0] - v2[0]) * (v4[1] - v3[1]) - (v3[1] - v2[1]) * (v4[0] - v3[0])
                if cross3 <= 1e-9:
                    continue
                
                # Turn at v4: (v3, v4, v5)
                cross4 = (v4[0] - v3[0]) * (v5[1] - v4[1]) - (v4[1] - v3[1]) * (v5[0] - v4[0])
                if cross4 <= 1e-9:
                    continue
                
                # Turn at v5: (v4, v5, v6)
                cross5 = (v5[0] - v4[0]) * (v6[1] - v5[1]) - (v5[1] - v4[1]) * (v6[0] - v5[0])
                if cross5 <= 1e-9:
                    continue
                
                # Turn at v6: (v5, v6, pivot)
                cross6 = (v6[0] - v5[0]) * (pivot[1] - v6[1]) - (v6[1] - v5[1]) * (pivot[0] - v6[0])
                if cross6 <= 1e-9:
                    continue
                
                # Turn at pivot: (v6, pivot, v1)
                cross7 = (pivot[0] - v6[0]) * (v1[1] - pivot[1]) - (pivot[1] - v6[1]) * (v1[0] - pivot[0])
                if cross7 <= 1e-9:
                    continue
                
                C += 1
        
        return C

    def objective(points_flat):
        """Objective function for optimization: minimize number of convex 7-gons."""
        points = points_flat.reshape(n, 2)
        if not is_valid(points):
            return 1e10
        C = count_convex7(points)
        return C

    # Start with a random valid configuration
    np.random.seed(42)
    best_points = None
    best_score = 1e18
    
    # Try several random starting points
    for seed in range(10):
        np.random.seed(seed)
        points = np.random.rand(n, 2)
        if is_valid(points):
            C = count_convex7(points)
            score = 1.0 / (1.0 + C)
            if best_points is None or score > best_score:
                best_score = score
                best_points = points.copy()
    
    if best_points is None:
        # Fallback: grid-like arrangement
        best_points = np.random.rand(n, 2)
        best_score = 1.0 / (1.0 + count_convex7(best_points))
    
    # Try to improve with local search
    start_time = time.time()
    step_size = 0.01
    for iteration in range(100000):
        if time.time() - start_time > 550:  # Safety margin
            break
        
        # Random perturbation
        mask = np.random.rand(n, 2) < 0.1
        perturbation = (np.random.rand(n, 2) - 0.5) * 2 * step_size * mask
        new_points = best_points + perturbation
        new_points = np.clip(new_points, 0, 1)
        
        if is_valid(new_points):
            C = count_convex7(new_points)
            score = 1.0 / (1.0 + C)
            if score > best_score:
                best_score = score
                best_points = new_points.copy()
    
    # Try differential evolution for a shorter time
    if time.time() - start_time < 500:
        try:
            bounds = [(0, 1)] * (2 * n)
            result = differential_evolution(
                objective, bounds, maxiter=50, popsize=15, seed=42,
                tol=1e-4, polish=False, init='latinhypercube'
            )
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(n, 2)
                if is_valid(candidate):
                    C = count_convex7(candidate)
                    score = 1.0 / (1.0 + C)
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
