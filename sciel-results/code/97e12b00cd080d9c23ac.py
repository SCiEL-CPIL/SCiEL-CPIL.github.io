# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 590  # wall-clock deadline with safety margin
_BEST_POINTS = None
_BEST_SCORE = 0.0  # 1/(1+C), higher is better (fewer convex 7-gons)


def _check_valid(points: np.ndarray) -> bool:
    """Validate according to the task contract."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    n = 33
    # Pairwise squared distances
    diff = points[:, None, :] - points[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, 1.0)
    if np.any(sq_dists < 1e-12):
        return False
    # Collinearity check: for all triples, |cross product| >= 1e-9
    # cross = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (points[i, 0] * (points[j, 1] - points[k, 1]) +
                         points[j, 0] * (points[k, 1] - points[i, 1]) +
                         points[k, 0] * (points[i, 1] - points[j, 1]))
                if abs(cross) < 1e-9:
                    return False
    return True


def _count_convex_7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position using the dynamic
    program over convex chains described in the task contract.
    
    Points are sorted lexicographically. For each pivot, later points are sorted
    by atan2 angle around it. A turn p -> q -> r counts as convex only if
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9, including closing turns back to pivot.
    """
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    total_convex_7 = 0
    
    for pivot in range(n):
        px, py = pts[pivot]
        # Later points (in lexicographic order)
        later = list(range(pivot + 1, n))
        if len(later) < 6:
            continue
        
        # Sort by atan2 angle around pivot
        def angle_key(idx):
            dx = pts[idx, 0] - px
            dy = pts[idx, 1] - py
            return np.arctan2(dy, dx)
        
        later.sort(key=angle_key)
        
        # Dynamic programming over convex chains
        # dp[i][j] = number of convex chains of length j ending at later[i]
        # where the chain starts from pivot and goes through later points in angular order
        # We need chains of length 6 (6 points after pivot to make 7 total including pivot)
        
        m = len(later)
        if m < 6:
            continue
        
        # dp[j][i] = number of ways to have a convex chain of j points (excluding pivot)
        # ending at later[i], where the chain uses points from later[0..i] in angular order
        # The chain must be convex: each consecutive triple makes a left turn
        
        # For a chain of length L (L points from 'later'), we need L=6 to form a 7-gon with pivot
        # The chain is: pivot -> later[i1] -> later[i2] -> ... -> later[i6] -> pivot
        # where i1 < i2 < ... < i6 in the angular-sorted order
        
        # dp[len][i] = number of convex chains of length 'len' ending at later[i]
        # Base: dp[1][i] = 1 for all i (chain of just later[i] from pivot)
        # Transition: dp[len][i] = sum over j < i where turn(later[j], later[i], next) is convex... 
        # Actually, we need to track the last two points to check convexity.
        
        # Let's use: dp[len][i][j] where i > j, meaning chain of length 'len' 
        # ending with ... -> later[j] -> later[i]
        # dp[2][i][j] = 1 if turn(pivot, later[j], later[i]) is convex (left turn)
        # dp[len][i][j] = sum over k < j where dp[len-1][j][k] > 0 and turn(later[k], later[j], later[i]) is convex
        
        # This is O(m^3 * 6) which for m up to 32 is manageable: 32^3 * 6 ~ 196k per pivot, 33 pivots ~ 6.5M
        
        # But we also need the closing turn: turn(later[i6], pivot, later[i1]) must be convex
        # This is tricky because the closing involves the first and last points of the chain.
        
        # Alternative: for each pivot, enumerate all 6-combinations of later points,
        # check if they form a convex polygon with the pivot.
        # C(32,6) = 906192 per pivot, 33 pivots ~ 30M checks. Each check is O(7) = 210M ops.
        # That's too slow in pure Python.
        
        # Let's use the DP approach more carefully.
        # For a fixed pivot, we want to count 6-subsets {i1 < i2 < ... < i6} from 'later'
        # such that the polygon pivot -> later[i1] -> ... -> later[i6] -> pivot is convex.
        
        # Convexity conditions:
        # 1. turn(pivot, later[i1], later[i2]) > 1e-9
        # 2. turn(later[i1], later[i2], later[i3]) > 1e-9
        # 3. turn(later[i2], later[i3], later[i4]) > 1e-9
        # 4. turn(later[i3], later[i4], later[i5]) > 1e-9
        # 5. turn(later[i4], later[i5], later[i6]) > 1e-9
        # 6. turn(later[i5], later[i6], pivot) > 1e-9
        # 7. turn(later[i6], pivot, later[i1]) > 1e-9
        
        # Precompute turn values
        # turn(a, b, c) = (bx-ax)(cy-by) - (by-ay)(cx-bx)
        
        # Let's precompute a turn matrix for the later points and pivot
        # indices: 0 = pivot, 1..m = later[0]..later[m-1]
        # turn[i][j][k] for i,j,k in 0..m
        
        # This is (m+1)^3 which for m=32 is 33^3 = 35937, fine.
        
        turn = np.zeros((m + 1, m + 1, m + 1))
        all_pts = np.vstack([pts[pivot:pivot+1], pts[later]])
        for a in range(m + 1):
            for b in range(m + 1):
                if a == b:
                    continue
                for c in range(m + 1):
                    if a == c or b == c:
                        continue
                    ax, ay = all_pts[a]
                    bx, by = all_pts[b]
                    cx, cy = all_pts[c]
                    turn[a, b, c] = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        
        # Now DP: dp[len][j][i] where 1 <= j < i <= m, meaning a convex chain of length 'len'
        # (len points from later, not counting pivot) ending at later[j-1] -> later[i-1]
        # (using 1-based indexing into later, so later[0] is index 1)
        
        # dp[2][j][i] = 1 if turn[0][j][i] > 1e-9 else 0
        # dp[len][j][i] = sum over k in [1, j-1] of dp[len-1][k][j] * (1 if turn[j][k]... wait
        
        # Let me re-index. Let later points be indexed 1..m.
        # A chain is a sequence 1 <= i1 < i2 < ... < i_len <= m
        # dp[len][i_{len-1}][i_len] = number of valid chains of length len ending with i_{len-1}, i_len
        
        # dp[2][i1][i2] = 1 if turn[0][i1][i2] > 1e-9
        # dp[len][i_{len-1}][i_len] = sum_{i_{len-2} < i_{len-1}} dp[len-1][i_{len-2}][i_{len-1}] * [turn[i_{len-2}][i_{len-1}][i_len] > 1e-9]
        
        # For len=6, we need to also check:
        # turn[i6][0][i1] > 1e-9  (closing turn back to pivot and to first point)
        # turn[i5][i6][0] > 1e-9  (this is already in the chain as turn[i5][i6][pivot])
        
        # Wait, the chain is pivot -> i1 -> i2 -> i3 -> i4 -> i5 -> i6 -> pivot
        # Turns to check:
        # turn[pivot][i1][i2] > 1e-9  (in dp[2])
        # turn[i1][i2][i3] > 1e-9  (in dp[3])
        # turn[i2][i3][i4] > 1e-9  (in dp[4])
        # turn[i3][i4][i5] > 1e-9  (in dp[5])
        # turn[i4][i5][i6] > 1e-9  (in dp[6])
        # turn[i5][i6][pivot] > 1e-9  (closing, need to check separately)
        # turn[i6][pivot][i1] > 1e-9  (closing, need to check separately)
        
        # So after computing dp[6][i5][i6], we need to sum over i1 < i2 < ... < i5 < i6
        # where dp[6][i5][i6] > 0 AND turn[i5][i6][0] > 1e-9 AND turn[i6][0][i1] > 1e-9
        
        # But dp[6][i5][i6] doesn't track i1. We need a different approach.
        
        # Let's just enumerate all 6-combinations for each pivot. C(32,6) = 906192
        # For each combination, check 7 turn conditions. That's 906192 * 7 = 6.3M per pivot.
        # 33 pivots = 208M. In Python this is too slow.
        
        # Let's use numpy vectorization for the turn checks.
        
        # Generate all 6-combinations of indices 0..m-1 (into 'later')
        # For m up to 32, C(32,6) = 906192
        # This is large but we can process in batches.
        
        # Actually, let's just do it with the DP but track the full chain.
        # Since len=6 is fixed, we can do:
        
        # For each i1 < i2 < i3 < i4 < i5 < i6:
        # Check all 7 turn conditions
        
        # Let's precompute which turns are valid:
        # valid_turn[a][b][c] = 1 if turn[a][b][c] > 1e-9 else 0
        
        valid = (turn > 1e-9).astype(np.int8)
        
        # Now for each 6-combination (i1,i2,i3,i4,i5,i6) from 1..m:
        # Check: valid[0][i1][i2] & valid[i1][i2][i3] & valid[i2][i3][i4] & 
        #        valid[i3][i4][i5] & valid[i4][i5][i6] & valid[i5][i6][0] & valid[i6][0][i1]
        
        # This is still O(C(m,6)) per pivot. Let's just do it.
        
        from itertools import combinations as comb
        for combo in comb(range(1, m + 1), 6):
            i1, i2, i3, i4, i5, i6 = combo
            if (valid[0, i1, i2] and valid[i1, i2, i3] and valid[i2, i3, i4] and
                    valid[i3, i4, i5] and valid[i4, i5, i6] and valid[i5, i6, 0] and
                    valid[i6, 0, i1]):
                total_convex_7 += 1
        
    return total_convex_7


def _evaluate(points: np.ndarray) -> float:
    """Compute 1/(1+C) where C is the number of convex 7-gons."""
    if not _check_valid(points):
        return 0.0
    C = _count_convex_7(points)
    return 1.0 / (1.0 + C)


def _perturb(points: np.ndarray, scale: float = 0.01) -> np.ndarray:
    """Apply small random perturbations to points."""
    new_points = points.copy()
    noise = np.random.randn(*new_points.shape) * scale
    new_points += noise
    # Clip to [0, 1]
    new_points = np.clip(new_points, 0.0, 1.0)
    return new_points


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    global _BEST_POINTS, _BEST_SCORE
    
    n = 33
    np.random.seed(42)
    
    # Start with a random configuration
    current_points = np.random.rand(n, 2)
    
    # Initial evaluation
    current_score = _evaluate(current_points)
    if current_score > _BEST_SCORE:
        _BEST_SCORE = current_score
        _BEST_POINTS = current_points.copy()
    
    # Local search with simulated annealing-like approach
    temp = 0.01
    min_temp = 0.0001
    cooling = 0.95
    
    while time.time() < _DEADLINE:
        # Try a perturbation
        new_points = _perturb(current_points, scale=temp)
        
        if not _check_valid(new_points):
            # Try to fix by re-randomizing problematic points
            continue
        
        new_score = _evaluate(new_points)
        
        if new_score >= current_score:
            current_points = new_points
            if new_score > _BEST_SCORE:
                _BEST_SCORE = new_score
                _BEST_POINTS = new_points.copy()
        else:
            # Accept with some probability (simulated annealing)
            if np.random.rand() < np.exp((new_score - current_score) / max(temp, 1e-10)):
                current_points = new_points
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
    
    # Return the best found
    if _BEST_POINTS is not None:
        return _BEST_POINTS
    return current_points


# EVOLVE-BLOCK-END
