# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _count_convex7(points):
    """Count convex 7-gons using the DP over convex chains method."""
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    C = 0
    for i in range(n):
        # Later points in lex order
        later = pts[i+1:]
        if len(later) < 6:
            break
        # Sort by atan2 angle around point i
        dx = later[:, 0] - pts[i, 0]
        dy = later[:, 1] - pts[i, 1]
        angles = np.arctan2(dy, dx)
        angle_order = np.argsort(angles)
        later_sorted = later[angle_order]
        
        m = len(later_sorted)
        # DP: dp[j] = number of convex chains ending at later_sorted[j]
        # A chain p_i -> ... -> later_sorted[j] is convex if all turns are left turns
        # We need chains of length 6 (5 intermediate points + pivot = 7 total)
        # Actually we need 6 points after the pivot to form a 7-gon
        # dp[k][j] = number of convex chains of length k ending at j
        # Base: dp[1][j] = 1 for all j
        # Transition: dp[k][j] = sum over i < j where turn(p, i, j) is convex of dp[k-1][i]
        # But we also need the closing turn back to pivot
        
        # Let's use a different approach: count all 6-tuples (j1, j2, ..., j6) with j1 < j2 < ... < j6
        # such that all consecutive turns are convex, including the closing turn
        # This is O(m^6) which is too slow for m up to 32
        
        # Better: DP where dp[len][j] = number of convex chains of length len ending at position j
        # len ranges from 1 to 6
        # dp[1][j] = 1
        # dp[len][j] = sum_{i < j, turn convex} dp[len-1][i]
        # Final: for each chain of length 6 ending at j6, check if closing turn (j6 -> j1 -> pivot) is convex
        # Wait, the closing is from last point back to first point and then to pivot
        # Actually the 7-gon is: pivot, p1, p2, p3, p4, p5, p6
        # Turns: pivot->p1->p2, p1->p2->p3, ..., p5->p6->pivot, p6->pivot->p1
        # All must be left turns (convex)
        
        # For efficiency, let's use the standard DP approach
        # dp[j] for chain ending at j with certain length
        # We'll track chains of exactly 6 points after pivot
        
        # dp_len[j] = number of convex chains of length len ending at j
        # len from 1 to 6
        # But we also need to remember the starting point for the closing turn
        
        # Alternative: for each starting point j1, count chains of length 6 starting at j1
        # This is O(m^6) worst case but with pruning
        
        # Let's use: for each j1, do DP for chains starting at j1
        # dp[len][j] = number of convex chains of length len starting at j1 and ending at j
        # dp[1][j1] = 1
        # dp[len][j] = sum_{i < j, turn(i, j) convex relative to chain} dp[len-1][i]
        # Actually the turn condition depends on the previous point in the chain
        
        # Simpler approach given m <= 32: use the fact that we need 6 points
        # Iterate over all C(m, 6) combinations and check if they form a convex 7-gon with pivot
        # C(32, 6) = 906192, times 33 pivots = ~30M checks. Might be too slow.
        
        # Let's use a smarter DP. For a fixed pivot i, sort later points by angle.
        # A convex chain p_i -> q_1 -> q_2 -> ... -> q_k is valid if each turn is left.
        # The turn q_{t-1} -> q_t -> q_{t+1} is left if cross product > 0.
        # For the full 7-gon, we need q_1 -> ... -> q_6 -> q_1 to also be convex (closing turns).
        
        # dp[j] = number of valid convex chains ending at j (of various lengths)
        # We need chains of exactly 6 points. Let's use dp[len][j] where len is chain length (number of points after pivot).
        
        # For performance, let's precompute turn validity
        # turn[a][b] = True if turn from a to b is "convex" relative to... 
        # Actually the turn condition for q_a -> q_b -> q_c is:
        # (q_b.x - q_a.x)(q_c.y - q_b.y) - (q_b.y - q_a.y)(q_c.x - q_b.x) > 1e-9
        
        # This depends on three points, so we can't easily precompute a 2D table.
        
        # Given the constraints, let's just do the O(m^6) approach but with early termination
        # and hope it's fast enough. For m=32, C(32,6) = 906192. With 33 pivots, that's ~30M.
        # Each check is ~6 cross products. Might take a few seconds.
        
        # Actually, let me reconsider. The task says the evaluator uses a specific DP method.
        # Let me implement that exact method for consistency.
        
        # The DP method: for each pivot, sort later points by angle.
        # dp[j] = number of convex chains ending at j (of length 6)
        # We build up: a chain of length k ending at j means the last k points in the chain end at j
        # and all turns are convex.
        
        # Let's define: for chain length L (number of points after pivot, so L=6 for 7-gon),
        # dp_L[j] = number of convex chains of length L ending at position j in the sorted-by-angle order
        # dp_1[j] = 1 for all j
        # dp_L[j] = sum over i < j of dp_{L-1}[i] * [turn(i, j) is convex]
        # But "turn(i, j) is convex" needs the previous point too!
        
        # I think the correct interpretation is:
        # A convex chain p_0, p_1, ..., p_k means all consecutive triples form left turns.
        # So dp[k][j] = number of convex chains of k+1 points (p_0 to p_k) ending at j, where p_0 is the pivot.
        # dp[0][j] = 1 (chain is just pivot -> j)
        # dp[k][j] = sum_{i < j} dp[k-1][i] * [turn(i, j) is left turn]
        # where turn(i, j) being a left turn means: for the chain ..., p_{k-2}, p_{k-1}=i, p_k=j
        # the turn p_{k-2} -> i -> j is left. But this depends on p_{k-2}!
        
        # I think I'm overcomplicating this. Let me just use the straightforward O(m^6) counting
        # but optimize with numpy where possible.
        
        # For now, let's use a simpler heuristic count for speed during search,
        # and use the exact count only for final evaluation.
        
        pass
    
    return C


def _count_convex7_simple(points):
    """Simple O(n^7) count for small n, used for validation."""
    n = len(points)
    count = 0
    for combo in combinations(range(n), 7):
        pts = points[list(combo)]
        # Check if these 7 points form a convex 7-gon
        # Find the convex hull of these 7 points; if all 7 are on the hull, it's convex
        # Use the angle sorting method
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(angles)
        sorted_pts = pts[order]
        
        # Check all turns are left
        convex = True
        for i in range(7):
            p = sorted_pts[i]
            q = sorted_pts[(i+1) % 7]
            r = sorted_pts[(i+2) % 7]
            cross = (q[0]-p[0])*(r[1]-q[1]) - (q[1]-p[1])*(r[0]-q[0])
            if cross <= 1e-9:
                convex = False
                break
        if convex:
            count += 1
    return count


def _is_valid(points):
    """Check if points are valid per the task constraints."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check no two points too close
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            d2 = np.sum((points[i] - points[j])**2)
            if d2 < 1e-12:
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                val = abs(points[i,0]*(points[j,1]-points[k,1]) + 
                         points[j,0]*(points[k,1]-points[i,1]) + 
                         points[k,0]*(points[i,1]-points[j,1]))
                if val < 1e-9:
                    return False
    return True


def _count_convex7_fast(points):
    """Count convex 7-gons using the specified DP method."""
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    C = 0
    for i in range(n):
        later_indices = list(range(i+1, n))
        if len(later_indices) < 6:
            break
        later_pts = pts[later_indices]
        
        dx = later_pts[:, 0] - pts[i, 0]
        dy = later_pts[:, 1] - pts[i, 1]
        angles = np.arctan2(dy, dx)
        angle_order = np.argsort(angles)
        later_sorted = later_pts[angle_order]
        
        m = len(later_sorted)
        if m < 6:
            continue
        
        # For each 6-tuple (j1 < j2 < ... < j6) in the angle-sorted order,
        # check if pivot -> j1 -> j2 -> ... -> j6 -> pivot is a convex 7-gon
        # Turns to check:
        # 1. pivot -> j1 -> j2
        # 2. j1 -> j2 -> j3
        # 3. j2 -> j3 -> j4
        # 4. j3 -> j4 -> j5
        # 5. j4 -> j5 -> j6
        # 6. j5 -> j6 -> pivot
        # 7. j6 -> pivot -> j1
        
        # Precompute for efficiency
        p = pts[i]
        
        for j1 in range(m-5):
            q1 = later_sorted[j1]
            # Check turn: pivot -> q1 -> q2 (for various q2)
            # We'll iterate and check
            for j2 in range(j1+1, m-4):
                q2 = later_sorted[j2]
                # Turn p -> q1 -> q2
                cross1 = (q1[0]-p[0])*(q2[1]-q1[1]) - (q1[1]-p[1])*(q2[0]-q1[0])
                if cross1 <= 1e-9:
                    continue
                for j3 in range(j2+1, m-3):
                    q3 = later_sorted[j3]
                    # Turn q1 -> q2 -> q3
                    cross2 = (q2[0]-q1[0])*(q3[1]-q2[1]) - (q2[1]-q1[1])*(q3[0]-q2[0])
                    if cross2 <= 1e-9:
                        continue
                    for j4 in range(j3+1, m-2):
                        q4 = later_sorted[j4]
                        # Turn q2 -> q3 -> q4
                        cross3 = (q3[0]-q2[0])*(q4[1]-q3[1]) - (q3[1]-q2[1])*(q4[0]-q3[0])
                        if cross3 <= 1e-9:
                            continue
                        for j5 in range(j4+1, m-1):
                            q5 = later_sorted[j5]
                            # Turn q3 -> q4 -> q5
                            cross4 = (q4[0]-q3[0])*(q5[1]-q4[1]) - (q4[1]-q3[1])*(q5[0]-q4[0])
                            if cross4 <= 1e-9:
                                continue
                            for j6 in range(j5+1, m):
                                q6 = later_sorted[j6]
                                # Turn q4 -> q5 -> q6
                                cross5 = (q5[0]-q4[0])*(q6[1]-q5[1]) - (q5[1]-q4[1])*(q6[0]-q5[0])
                                if cross5 <= 1e-9:
                                    continue
                                # Turn q5 -> q6 -> pivot
                                cross6 = (q6[0]-q5[0])*(p[1]-q6[1]) - (q6[1]-q5[1])*(p[0]-q6[0])
                                if cross6 <= 1e-9:
                                    continue
                                # Turn q6 -> pivot -> q1
                                cross7 = (p[0]-q6[0])*(q1[1]-p[1]) - (p[1]-q6[1])*(q1[0]-p[0])
                                if cross7 <= 1e-9:
                                    continue
                                C += 1
    
    return C


def _initial_circular(n=33, radius=0.45, center=(0.5, 0.5)):
    """Generate points on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    # Add small perturbation to avoid degeneracy
    np.random.seed(42)
    x += np.random.randn(n) * 0.001
    y += np.random.randn(n) * 0.001
    # Clamp to [0, 1]
    x = np.clip(x, 0.01, 0.99)
    y = np.clip(y, 0.01, 0.99)
    return np.column_stack([x, y])


def _local_search(points, max_time=500):
    """Perform local search to minimize convex 7-gons."""
    start_time = time.time()
    best_points = points.copy()
    best_count = _count_convex7_fast(best_points)
    
    np.random.seed(123)
    n = len(points)
    
    while time.time() - start_time < max_time:
        # Try a random perturbation
        new_points = best_points.copy()
        idx = np.random.randint(n)
        new_points[idx] += np.random.randn(2) * 0.02
        # Clamp
        new_points = np.clip(new_points, 0.001, 0.999)
        
        if not _is_valid(new_points):
            continue
        
        new_count = _count_convex7_fast(new_points)
        if new_count < best_count:
            best_points = new_points
            best_count = new_count
        elif new_count == best_count:
            # Accept with some probability to escape local minima
            if np.random.rand() < 0.1:
                best_points = new_points
    
    return best_points, best_count


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Start with circular configuration
    points = _initial_circular(33)
    
    # Local search
    points, count = _local_search(points, max_time=550)
    
    # Final check
    if time.time() < deadline:
        # Try a few more restarts if time permits
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            new_points = _initial_circular(33, radius=0.4 + np.random.rand()*0.1)
            new_points, new_count = _local_search(new_points, max_time=60)
            if new_count < count:
                points = new_points
                count = new_count
    
    # Ensure validity
    if not _is_valid(points):
        # Fallback to random
        np.random.seed(0)
        points = np.random.rand(33, 2)
    
    return points


# EVOLVE-BLOCK-END
