# EVOLVE-BLOCK-START
import numpy as np
import time
from math import atan2, pi

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position.
    Uses the standard dynamic programming approach over convex chains.
    """
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    total = 0
    
    for i in range(n - 6):
        # Points after i
        later = pts[i+1:]
        if len(later) < 6:
            break
        
        # Compute angles from i to each later point
        dx = later[:, 0] - pts[i, 0]
        dy = later[:, 1] - pts[i, 1]
        angles = np.arctan2(dy, dx)
        
        # Sort by angle
        angle_order = np.argsort(angles)
        sorted_pts = later[angle_order]
        m = len(sorted_pts)
        
        # dp[l][b] = number of convex chains of length l ending at position b
        # where the chain starts at some position a < b
        # We want chains of length 6 (6 points from sorted_pts)
        
        # Base: chains of length 1
        dp1 = np.ones(m, dtype=np.int64)
        
        # For length 2: dp2[b] = sum over a < b of [turn i -> sorted_pts[a] -> sorted_pts[b] is convex]
        # Actually, let's be more careful.
        
        # A convex chain of length l: q1, q2, ..., ql where q1 is at position a1, q2 at a2, ..., ql at al
        # with a1 < a2 < ... < al
        # The turns are: i->q1->q2, q1->q2->q3, ..., q_{l-2}->q_{l-1}->q_l
        # All must be convex (positive cross product)
        
        # For the 7-gon, we need l=6, and also the closing turn q6->i must be convex.
        
        # Let's compute this step by step.
        
        # Precompute cross products for efficiency
        # cross(a, b, c) = (b-a) x (c-a) = (bx-ax)(cy-ay) - (by-ay)(cx-ax)
        
        # For turn i -> q_a -> q_b: cross = (q_a - i) x (q_b - i)
        # Actually: (qax - ix)(qby - i y) - (qay - iy)(qbx - ix)
        
        # Let's use a simpler approach: for small m (up to 32), we can enumerate.
        # But C(32,6) = 906192, and we have 33 pivots, so ~30M checks. Each check is O(6).
        # That's ~180M operations, which might be too slow in Python.
        
        # Let's use the DP approach properly.
        
        # dp[l][b] = number of convex chains of length l ending at position b
        # A chain of length l ending at b: q1, q2, ..., ql where ql is at position b
        # and q1 is at some position a1 < a2 < ... < al = b
        # The turns must all be convex.
        
        # dp[1][b] = 1 for all b (chain of just one point)
        # dp[2][b] = number of a < b such that turn i -> q_a -> q_b is convex
        # dp[l][b] = sum over a < b of dp[l-1][a] * [turn q_{l-2} -> q_a -> q_b is convex]
        
        # Wait, this isn't quite right. The turn depends on the previous point in the chain.
        
        # Let me reconsider. Let's define:
        # dp[l][b] = number of convex chains of length l ending at position b,
        # where the chain is q1, q2, ..., ql with positions a1 < a2 < ... < al = b,
        # and all turns i->q1->q2, q1->q2->q3, ..., q_{l-2}->q_{l-1}->q_l are convex.
        
        # For l=1: dp[1][b] = 1
        # For l=2: dp[2][b] = sum over a < b of [turn i -> q_a -> q_b is convex]
        #   where turn i -> q_a -> q_b convex means (q_a - i) x (q_b - i) > 0
        #   Actually, the turn at q_a: (q_a - i) x (q_b - q_a) > 0
        #   Let's use: cross(i, q_a, q_b) = (q_a[0]-i[0])*(q_b[1]-q_a[1]) - (q_a[1]-i[1])*(q_b[0]-q_a[0])
        
        # For l>=3: dp[l][b] = sum over a < b of dp[l-1][a] * [turn q_{l-2} -> q_a -> q_b is convex]
        #   But we don't track q_{l-2}. This is the issue.
        
        # Alternative: let's just enumerate all 6-subsets for each pivot.
        # m is at most 32, so C(32,6) = 906192. For 33 pivots, that's ~30M.
        # Each check is O(6) cross products. So ~180M cross product evaluations.
        # In Python, this might take a few minutes. Let's try to optimize.
        
        # Actually, let's use a different approach. Let's precompute the cross products.
        
        # For each pivot i, and for each pair (a, b) with a < b in the sorted order,
        # compute whether turn i -> q_a -> q_b is convex.
        
        # Then, for triples (a, b, c) with a < b < c, check turn q_a -> q_b -> q_c.
        
        # This is getting complex. Let's just do the enumeration but optimize the inner loop.
        
        # Precompute cross products for all pairs
        # cross_ab = cross product of (q_a - i) and (q_b - i) for turn i->a->b
        # Actually, for the turn at a: (q_a - i) x (q_b - q_a)
        
        # Let's just enumerate.
        from itertools import combinations
        
        for subset in combinations(range(m), 6):
            # Check if i -> q_{s0} -> q_{s1} -> ... -> q_{s5} -> i is convex
            valid = True
            # Turn at q_{s0}: i -> q_{s0} -> q_{s1}
            a, b = subset[0], subset[1]
            cross = (sorted_pts[a,0]-pts[i,0])*(sorted_pts[b,1]-sorted_pts[a,1]) - \
                    (sorted_pts[a,1]-pts[i,1])*(sorted_pts[b,0]-sorted_pts[a,0])
            if cross <= 1e-9:
                valid = False
            else:
                # Turn at q_{s1}: q_{s0} -> q_{s1} -> q_{s2}
                a, b = subset[0], subset[1]
                c = subset[2]
                cross = (sorted_pts[b,0]-sorted_pts[a,0])*(sorted_pts[c,1]-sorted_pts[b,1]) - \
                        (sorted_pts[b,1]-sorted_pts[a,1])*(sorted_pts[c,0]-sorted_pts[b,0])
                if cross <= 1e-9:
                    valid = False
                else:
                    # Turn at q_{s2}: q_{s1} -> q_{s2} -> q_{s3}
                    a, b = subset[1], subset[2]
                    c = subset[3]
                    cross = (sorted_pts[b,0]-sorted_pts[a,0])*(sorted_pts[c,1]-sorted_pts[b,1]) - \
                            (sorted_pts[b,1]-sorted_pts[a,1])*(sorted_pts[c,0]-sorted_pts[b,0])
                    if cross <= 1e-9:
                        valid = False
                    else:
                        # Turn at q_{s3}: q_{s2} -> q_{s3} -> q_{s4}
                        a, b = subset[2], subset[3]
                        c = subset[4]
                        cross = (sorted_pts[b,0]-sorted_pts[a,0])*(sorted_pts[c,1]-sorted_pts[b,1]) - \
                                (sorted_pts[b,1]-sorted_pts[a,1])*(sorted_pts[c,0]-sorted_pts[b,0])
                        if cross <= 1e-9:
                            valid = False
                        else:
                            # Turn at q_{s4}: q_{s3} -> q_{s4} -> q_{s5}
                            a, b = subset[3], subset[4]
                            c = subset[5]
                            cross = (sorted_pts[b,0]-sorted_pts[a,0])*(sorted_pts[c,1]-sorted_pts[b,1]) - \
                                    (sorted_pts[b,1]-sorted_pts[a,1])*(sorted_pts[c,0]-sorted_pts[b,0])
                            if cross <= 1e-9:
                                valid = False
                            else:
                                # Closing turn: q_{s5} -> i
                                # Turn at q_{s5}: q_{s4} -> q_{s5} -> i
                                a, b = subset[4], subset[5]
                                cross = (sorted_pts[b,0]-sorted_pts[a,0])*(pts[i,1]-sorted_pts[b,1]) - \
                                        (sorted_pts[b,1]-sorted_pts[a,1])*(pts[i,0]-sorted_pts[b,0])
                                if cross <= 1e-9:
                                    valid = False
            
            if valid:
                total += 1
    
    return total

def _is_valid(points: np.ndarray) -> bool:
    """Check if the point set is valid."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(points >= -1e-6) or not np.all(points <= 1 + 1e-6):
        return False
    
    # Check no two points too close
    for i in range(33):
        for j in range(i+1, 33):
            d2 = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2
            if d2 < 1e-12:
                return False
    
    # Check no three collinear
    for i in range(33):
        for j in range(i+1, 33):
            for k in range(j+1, 33):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                if abs(cross) < 1e-9:
                    return False
    
    return True

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin
    
    # Initial configuration: cluster points in corners to reduce convex polygons
    np.random.seed(42)
    
    # Start with points clustered in 4 corners
    points = np.zeros((n, 2))
    corner_centers = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    sigma = 0.03
    
    for i in range(n):
        corner = corner_centers[i % 4]
        points[i, 0] = corner[0] + np.random.normal(0, sigma)
        points[i, 1] = corner[1] + np.random.normal(0, sigma)
    
    # Clip to valid range
    points = np.clip(points, 0.02, 0.98)
    
    # Ensure validity
    if not _is_valid(points):
        perturb = np.random.uniform(-0.005, 0.005, size=(n, 2))
        points = points + perturb
        points = np.clip(points, 0.02, 0.98)
    
    best_points = points.copy()
    best_count = _count_convex7(points)
    
    # Hill climbing with restarts
    step_size = 0.02
    no_improve = 0
    max_no_improve = 200
    restart_count = 0
    max_restarts = 5
    
    while time.time() < deadline:
        # Try moving a random point
        idx = np.random.randint(n)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        new_points = np.clip(new_points, 0.005, 0.995)
        
        if not _is_valid(new_points):
            no_improve += 1
            continue
        
        count = _count_convex7(new_points)
        
        if count < best_count:
            best_points = new_points.copy()
            best_count = count
            points = new_points.copy()
            no_improve = 0
            step_size *= 0.97
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                step_size = min(step_size * 1.3, 0.05)
        
        if no_improve >= max_no_improve:
            # Restart from best with new random perturbations
            restart_count += 1
            if restart_count > max_restarts:
                break
            points = best_points + np.random.normal(0, 0.02, size=(n, 2))
            points = np.clip(points, 0.02, 0.98)
            if not _is_valid(points):
                points = best_points.copy()
            no_improve = 0
            step_size = 0.02
    
    return best_points

# EVOLVE-BLOCK-END
