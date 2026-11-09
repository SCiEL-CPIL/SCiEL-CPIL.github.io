# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from math import atan2, pi

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position using a dynamic program
    over convex chains.
    
    Points are sorted lexicographically. For each pivot i, later points (j > i) are
    sorted by atan2 angle around i. A turn p->q->r counts as convex if
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    
    We count 7-gons by: for each 7-tuple of points in convex position, the "smallest"
    point in lexicographic order is the pivot, and the remaining 6 form a convex chain.
    Actually, we need to be more careful. Let's use the standard approach:
    
    For each pair (i, j) with i < j, we count the number of convex chains of length L
    starting with i, j (where all subsequent points have index > j). A 7-gon is counted
    when we have a convex chain of 6 points (i, j, p3, p4, p5, p6) where the turn
    from p6 back to i is also convex.
    """
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    total = 0
    
    # For each pivot i, consider points j > i
    for i in range(n - 6):
        # Get points after i
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
        
        # For each pair (a, b) with a < b in the sorted order,
        # count convex chains starting with i -> sorted_pts[a] -> sorted_pts[b]
        # and extending with points after b
        
        # DP: dp[k][b] = number of convex chains of length k ending at position b
        # where the chain starts with i, sorted_pts[a] for some a < b
        # We want chains of total length 6 (i + 5 more points)
        
        # Actually, let's use a different formulation:
        # A convex 7-gon with pivot i means: i, p2, p3, p4, p5, p6, p7 form a convex polygon
        # where i is the lexicographically smallest, and p2..p7 are in angular order around i.
        # The turns must all be convex.
        
        # Let's count: for each choice of 6 points from sorted_pts, in their angular order,
        # check if i -> p1 -> p2 -> ... -> p6 -> i forms a convex chain.
        
        # This is O(m choose 6) which is too slow for large m.
        
        # Better: DP over positions.
        # dp[length][pos] = number of convex chains of 'length' points (not counting i)
        # ending at position 'pos' in sorted_pts, where all points have index < pos.
        
        # We want to count chains of length 6 (6 points from sorted_pts) such that
        # the turn from the last point back to i is also convex.
        
        # Let's define:
        # dp[l][b] = number of convex chains of length l (l points from sorted_pts, not counting i)
        # ending at position b, where the chain starts at some position a < b,
        # and all turns are convex.
        
        # Base case: dp[1][b] = 1 for all b (chain of just sorted_pts[b])
        # Wait, we need at least 2 points to form a turn. Let me reconsider.
        
        # A chain of length l means l points from sorted_pts: q1, q2, ..., ql
        # where q1 is the first (smallest angle), ql is the last.
        # The turns are: i->q1->q2, q1->q2->q3, ..., q_{l-2}->q_{l-1}->q_l
        # For the 7-gon, we need l=6, and also the turn q_l->i must be convex (closing).
        
        # dp[l][b] = number of ways to choose a chain of l points ending at position b
        # dp[1][b] = 1 (just the point at b)
        # dp[l][b] = sum over a < b of dp[l-1][a] * [turn a->b is convex relative to previous]
        
        # Hmm, this is getting complicated. Let me use a simpler but still efficient approach.
        
        # For small m, we can enumerate. For m up to ~30, C(m,6) is manageable.
        # m is at most 32, C(32,6) = 906192, times 33 pivots = ~30M, which might be okay.
        
        # Actually, let's just do the direct count for clarity and correctness.
        for subset in itertools.combinations(range(m), 6):
            # Check if i -> sorted_pts[subset[0]] -> ... -> sorted_pts[subset[5]] -> i is convex
            valid = True
            prev = pts[i]
            for idx in subset:
                curr = sorted_pts[idx]
                if idx == subset[0]:
                    # First turn: i -> curr -> next
                    if len(subset) > 1:
                        nxt = sorted_pts[subset[1]]
                        cross = (curr[0]-prev[0])*(nxt[1]-curr[1]) - (curr[1]-prev[1])*(nxt[0]-curr[0])
                        if cross <= 1e-9:
                            valid = False
                            break
                else:
                    # Turn from previous point in subset to curr to next
                    prev_idx = subset[subset.index(idx)-1]
                    prev_pt = sorted_pts[prev_idx]
                    if idx < subset[5]:
                        nxt_idx = subset[subset.index(idx)+1]
                        nxt_pt = sorted_pts[nxt_idx]
                        cross = (curr[0]-prev_pt[0])*(nxt_pt[1]-curr[1]) - (curr[1]-prev_pt[1])*(nxt_pt[0]-curr[0])
                        if cross <= 1e-9:
                            valid = False
                            break
                    else:
                        # Last point, check closing turn: curr -> i
                        # The turn is prev_pt -> curr -> i
                        cross = (curr[0]-prev_pt[0])*(pts[i,1]-curr[1]) - (curr[1]-prev_pt[1])*(pts[i,0]-curr[0])
                        if cross <= 1e-9:
                            valid = False
                            break
                prev = curr
            
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
    
    Uses hill-climbing with a fast convex 7-gon counter.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 600  # 10 minute deadline
    
    # Initial configuration: cluster points in corners to reduce convex polygons
    np.random.seed(42)
    
    # Start with points clustered in 4 corners
    points = np.zeros((n, 2))
    corner_centers = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    sigma = 0.02
    
    for i in range(n):
        corner = corner_centers[i % 4]
        points[i, 0] = corner[0] + np.random.normal(0, sigma)
        points[i, 1] = corner[1] + np.random.normal(0, sigma)
    
    # Clip to valid range
    points = np.clip(points, 0.01, 0.99)
    
    # Ensure validity
    if not _is_valid(points):
        # Add small perturbations
        perturb = np.random.uniform(-0.001, 0.001, size=(n, 2))
        points = points + perturb
        points = np.clip(points, 0.01, 0.99)
    
    best_points = points.copy()
    best_count = _count_convex7(points)
    
    # Hill climbing
    step_size = 0.01
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try moving a random point
        idx = np.random.randint(n)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        new_points = np.clip(new_points, 0.001, 0.999)
        
        if not _is_valid(new_points):
            no_improve += 1
            continue
        
        count = _count_convex7(new_points)
        
        if count < best_count:
            best_points = new_points.copy()
            best_count = count
            points = new_points.copy()
            no_improve = 0
            # Reduce step size on improvement
            step_size *= 0.95
        else:
            no_improve += 1
            # Occasionally increase step size to escape local minima
            if no_improve % 50 == 0:
                step_size = min(step_size * 1.5, 0.05)
    
    return best_points

# EVOLVE-BLOCK-END
