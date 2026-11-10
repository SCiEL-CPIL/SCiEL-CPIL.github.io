# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position.
    
    Uses the dynamic programming approach:
    - Sort points lexicographically
    - For each pivot i, consider points j > i sorted by angle around i
    - Count convex chains of length 6 (forming 7-gons with the pivot)
    """
    n = len(points)
    if n < 7:
        return 0
    
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    total = 0
    
    for i in range(n - 6):
        # Points after i
        later = pts[i+1:]
        if len(later) < 6:
            break
        
        # Sort by angle around pts[i]
        dx = later[:, 0] - pts[i, 0]
        dy = later[:, 1] - pts[i, 1]
        angles = np.arctan2(dy, dx)
        angle_order = np.argsort(angles)
        sorted_later = later[angle_order]
        m = len(sorted_later)
        
        # DP: dp[l][j] = number of convex chains of length l ending at position j
        # in sorted_later, where all points are after i
        # We want chains of length 6 (6 points + pivot = 7-gon)
        
        # For a chain to be convex, each turn must be a left turn
        # Turn from a to b to c is convex if (b-a) x (c-b) > 0, i.e.,
        # (bx-ax)(cy-by) - (by-ay)(cx-bx) > 0
        
        # dp[j] for length 2: just 1 (any pair forms a chain of length 2)
        # Actually, let's think more carefully.
        
        # A convex chain of length k ending at point j (in sorted_later)
        # means k-1 points before j in the chain, all with smaller angle,
        # and each consecutive triple makes a left turn.
        
        # Let's use: dp[l][j] = number of convex chains of length l ending at sorted_later[j]
        # Base: dp[1][j] = 1 for all j
        # dp[2][j] = j (any point before j can be the previous one) -- wait, no
        # Actually for length 2, we just need one previous point, so dp[2][j] = j
        # For length l >= 3: dp[l][j] = sum over k < j where turn (prev, sorted_later[k], sorted_later[j]) is convex
        # But we need to track what the previous point was...
        
        # Standard approach: dp[l][j] = number of convex chains of length l ending at j
        # where the chain is in increasing angle order.
        # dp[1][j] = 1
        # dp[2][j] = j  (any of the j points before can be the first)
        # dp[l][j] = sum_{k<j} dp[l-1][k] * [turn is convex]
        # where the turn is from the point before k in the chain to k to j.
        # But we don't know what the point before k is...
        
        # Actually, the standard ES counting uses:
        # For each pair (i, j) with i < j in the angular ordering, count chains.
        # Let's use a different formulation.
        
        # f[a][b] = number of convex chains starting with the pair (i, sorted_later[a], sorted_later[b])
        # where a < b, and extending with points after b.
        
        # Actually, let me use the simpler O(n^3) or O(n^4) approach for small n.
        # With m up to 32, O(m^4) might be too slow. Let's use the DP properly.
        
        # The key insight: a convex chain in angular order around pivot i
        # is a sequence of points p1, p2, ..., pk (in increasing angle order)
        # such that for each consecutive triple, the turn is to the left.
        # The turn from p_{t-1} to p_t to p_{t+1} is left if
        # (p_t - p_{t-1}) x (p_{t+1} - p_t) > 0
        
        # dp[l][j] = number of convex chains of length l ending at position j
        # dp[1][j] = 1
        # dp[2][j] = j  (choose any one point before j as the start)
        # Wait, that's not right either. Let me think again.
        
        # Actually, for a chain of length 2 ending at j, the previous point can be any k < j.
        # So dp[2][j] = j.
        
        # For dp[l][j] with l >= 3: we need to sum over all k < j such that
        # the turn from (the point before k in the chain) to k to j is convex.
        # But we don't know what the point before k is.
        
        # The correct formulation:
        # dp[l][j] = sum_{k=0}^{j-1} dp[l-1][k] * I(turn(k, j) is convex given the previous point)
        
        # This is tricky because the convexity depends on the previous point.
        
        # Alternative: use the fact that we can precompute for each pair (k, j)
        # whether adding j after k is valid, but this depends on the point before k.
        
        # Let me use a different approach: enumerate all 6-tuples in angular order
        # and check if they form a convex chain. With m up to ~32, C(32,6) is about 906192,
        # which is manageable.
        
        # Actually, let's use the DP with state (last_two_points).
        # dp[j1][j2] = number of convex chains ending with points at positions j1, j2 (j1 < j2)
        # Then to extend: for j3 > j2, if turn(j1, j2, j3) is convex, add to dp[j2][j3]
        
        # Base: dp[j1][j2] = 1 for all j1 < j2 (chain of length 2)
        # We want chains of length 6, so we need to extend 4 more times.
        
        # Let's use: dp_len[l][j1][j2] where l is the length so far
        # This gets complex. Let me just use the brute force for 6-tuples since m <= 32.
        
        # For each combination of 6 points from sorted_later (in angular order),
        # check if they form a convex chain with the pivot.
        
        # A 7-gon with pivot i and 6 other points in angular order p1...p6 is convex
        # if all turns are left:
        # turn(i, p1, p2), turn(p1, p2, p3), ..., turn(p5, p6, i) are all left turns.
        
        # Wait, the problem says "including the closing turns back to the pivot".
        # So we need: turn(i, p1, p2), turn(p1, p2, p3), ..., turn(p4, p5, p6),
        # turn(p5, p6, i), and turn(p6, i, p1) are all convex (left turns).
        
        # Actually, for a convex polygon, all turns should be in the same direction.
        # If we go around in angular order around the pivot, the turns should all be left
        # (assuming counterclockwise orientation).
        
        # Let me just check: for points i, p1, p2, ..., p6 in angular order around i,
        # the 7-gon is convex if:
        # 1. turn(i, p1, p2) > 0
        # 2. turn(p1, p2, p3) > 0
        # 3. turn(p2, p3, p4) > 0
        # 4. turn(p3, p4, p5) > 0
        # 5. turn(p4, p5, p6) > 0
        # 6. turn(p5, p6, i) > 0
        # 7. turn(p6, i, p1) > 0
        
        # where turn(a, b, c) = (bx-ax)(cy-by) - (by-ay)(cx-bx)
        
        # Let me implement this with itertools.combinations for the 6 points.
        
        from itertools import combinations
        
        for combo in combinations(range(m), 6):
            p_idx = [sorted_later[k] for k in combo]
            p0 = pts[i]
            p1, p2, p3, p4, p5, p6 = p_idx
            
            # Check all 7 turns
            # turn(p0, p1, p2)
            t1 = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
            if t1 <= 1e-9:
                continue
            # turn(p1, p2, p3)
            t2 = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
            if t2 <= 1e-9:
                continue
            # turn(p2, p3, p4)
            t3 = (p3[0]-p2[0])*(p4[1]-p3[1]) - (p3[1]-p2[1])*(p4[0]-p3[0])
            if t3 <= 1e-9:
                continue
            # turn(p3, p4, p5)
            t4 = (p4[0]-p3[0])*(p5[1]-p4[1]) - (p4[1]-p3[1])*(p5[0]-p4[0])
            if t4 <= 1e-9:
                continue
            # turn(p4, p5, p6)
            t5 = (p5[0]-p4[0])*(p6[1]-p5[1]) - (p5[1]-p4[1])*(p6[0]-p5[0])
            if t5 <= 1e-9:
                continue
            # turn(p5, p6, p0)
            t6 = (p6[0]-p5[0])*(p0[1]-p6[1]) - (p6[1]-p5[1])*(p0[0]-p6[0])
            if t6 <= 1e-9:
                continue
            # turn(p6, p0, p1)
            t7 = (p0[0]-p6[0])*(p1[1]-p0[1]) - (p0[1]-p6[1])*(p1[0]-p0[0])
            if t7 <= 1e-9:
                continue
            
            total += 1
    
    return total


def _is_valid(points: np.ndarray) -> bool:
    """Check if the point configuration is valid."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    
    # Check pairwise distances
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            d2 = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2
            if d2 < 1e-12:
                return False
    
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = abs(points[i,0]*(points[j,1]-points[k,1]) + 
                          points[j,0]*(points[k,1]-points[i,1]) + 
                          points[k,0]*(points[i,1]-points[j,1]))
                if area < 1e-9:
                    return False
    
    return True


def _generate_initial_points() -> np.ndarray:
    """Generate initial points using a structured approach to minimize convex 7-gons."""
    np.random.seed(12345)
    
    # Strategy: cluster points near the corners and edges to limit convex chains
    # Use a mix of corner clusters and edge points
    
    points = []
    
    # Corner clusters - 4 corners with multiple points each
    corners = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    per_corner = 8  # 4 * 8 = 32
    
    for cx, cy in corners:
        for i in range(per_corner):
            # Small random offset around the corner
            angle = 2 * np.pi * i / per_corner
            r = 0.05 + np.random.uniform(0, 0.03)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            # Clip to [0, 1]
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            points.append((x, y))
    
    # One point in the center
    points.append((0.5 + np.random.uniform(-0.05, 0.05), 
                   0.5 + np.random.uniform(-0.05, 0.05)))
    
    points = np.array(points)
    
    # Add small perturbations to ensure general position
    perturb = np.random.uniform(-0.002, 0.002, size=(33, 2))
    points = points + perturb
    points = np.clip(points, 0.001, 0.999)
    
    return points


def _hill_climb(points: np.ndarray, max_time: float = 500.0) -> np.ndarray:
    """Hill climbing optimization to reduce convex 7-gon count."""
    start_time = time.time()
    current_points = points.copy()
    current_count = _count_convex7(current_points)
    best_points = current_points.copy()
    best_count = current_count
    
    step_size = 0.02
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        # Try a random perturbation
        idx = np.random.randint(0, 33)
        delta = np.random.uniform(-step_size, step_size, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Check bounds
        if np.any(new_points < 0) or np.any(new_points > 1):
            continue
        
        # Check validity (quick check)
        if not _is_valid(new_points):
            continue
        
        new_count = _count_convex7(new_points)
        
        if new_count < current_count:
            current_points = new_points
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_points = new_points.copy()
            # Reduce step size slightly when improving
            step_size = max(0.001, step_size * 0.95)
        else:
            # Occasionally accept worse solutions to escape local minima
            if np.random.random() < 0.05:
                current_points = new_points
                current_count = new_count
            step_size = min(0.05, step_size * 1.01)
    
    return best_points


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill
    
    # Generate initial points
    points = _generate_initial_points()
    
    # Ensure validity
    if not _is_valid(points):
        # Try different seed
        np.random.seed(99999)
        points = _generate_initial_points()
    
    # Hill climb with time budget
    remaining_time = deadline - (time.time() - start_time) - 10.0  # 10s safety margin
    if remaining_time > 30:
        points = _hill_climb(points, max_time=remaining_time)
    
    # Final validation
    if not _is_valid(points):
        # Fallback to a simple valid configuration
        np.random.seed(777)
        points = np.random.uniform(0.05, 0.95, size=(33, 2))
        # Ensure no two points are too close
        for i in range(33):
            for j in range(i+1, 33):
                d2 = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2
                if d2 < 1e-12:
                    points[j, 0] += 0.01
                    points[j, 1] += 0.01
    
    return points

# EVOLVE-BLOCK-END
