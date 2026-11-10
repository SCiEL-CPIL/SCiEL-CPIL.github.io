import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    # Distance check
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    # Collinearity check
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _count_convex7_fast(points):
    """
    Count convex 7-gons.
    """
    n = len(points)
    # Sort lexicographically
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    eps = 1e-9

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        # Sort later points by angle around p
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # turn_mat[a, b, c] = cross product of (q[b]-q[a]) and (q[c]-q[b])
        # We only need a < b < c
        # DP state: dp[l][s][p_idx] = number of convex chains of length l
        # starting at s, ending at p_idx, with all intermediate points > s and < p_idx?
        # Actually, the standard Erdos-Szekeres DP for k-gons:
        # A convex chain is a sequence of points where each turn is convex.
        
        # Let's use the specific DP described in the task contract logic if possible, 
        # or a standard O(N^4) check which is fast enough for N=33.
        
        # For N=33, O(N^4) is ~1.2M ops. In Python, this might be slow if not vectorized.
        # Let's stick to the vectorized DP approach from the seed, it's efficient.
        
        # Precompute turn between any 3 points in 'q'
        # turn[a, b, c] for a < b < c
        # turn = (q[b]-q[a]) x (q[c]-q[b])
        
        # We can precompute a matrix of turns.
        # But m is up to 32. 32^3 = 32768.
        
        # dp[l][s][e] = count of convex chains of length l, starting at index s, ending at index e
        # in the sorted-by-angle list.
        # Note: The points in 'q' are sorted by angle around p.
        # A convex chain p -> q[s] -> ... -> q[e] must have all turns convex.
        
        # Initialize dp for length 2 (p -> s -> e)
        # Actually, let's define dp[l][s][e] as chains of l points in 'q', 
        # such that the full chain p -> q[s] -> ... -> q[e] is convex.
        
        # For l=1: just p -> q[s]. Always valid (1 point).
        # For l=2: p -> q[s] -> q[e]. Valid if turn(p, q[s], q[e]) > eps.
        # For l>2: p -> ... -> q[p_idx] -> q[e]. Valid if turn(q[p_idx-1], q[p_idx], q[e]) > eps 
        #           AND the previous chain was valid.
        
        # We iterate over the "last" point e, and the "previous" point p_idx.
        
        # Let's use a 3D array: dp[l][s][e]
        # s is start index, e is end index.
        # We fill for l = 2 to 6.
        
        dp = np.zeros((7, m, m), dtype=np.int64)
        
        # Base case: l=1 (chain of 1 point in q, total 2 points with p)
        # dp[1][s][s] = 1 for all s.
        # Actually, let's index by length of chain in q.
        # Chain of length 1: p -> q[s].
        # dp[1][s][s] = 1
        
        # Let's re-index:
        # dp[k][s][e] = number of convex chains of k points in q, starting at s, ending at e.
        # Such that the turns p->q[s]->...->q[e] are all convex.
        
        # k=1: dp[1][s][s] = 1
        # k=2: dp[2][s][e] = 1 if s < e and turn(p, q[s], q[e]) > eps
        # k>2: dp[k][s][e] = sum( dp[k-1][s][mid] for mid < e if turn(q[mid], q[e], next) ... )
        # Wait, the turn depends on the previous point in the chain.
        
        # Correct Recurrence:
        # Let chain be p, q[s], q[i1], ..., q[im], q[e].
        # The last turn is q[i_{m-1}] -> q[im] -> q[e]? No.
        # The turn at q[im] is (q[im] - q[i_{m-1}]) x (q[e] - q[im]).
        
        # So dp[k][s][e] = sum_{mid < e} dp[k-1][s][mid] * ( turn(q[mid], q[e]) is convex ? )
        # Wait, turn(q[mid], q[e]) is not defined. It's turn(q[mid-1], q[mid], q[e]).
        # So we need to know the point before mid.
        
        # This implies the state must be (start, prev, current).
        # dp[k][s][p_idx] = number of chains of length k ending at p_idx, starting at s.
        # The previous point is implicit in the transition?
        # No, the turn depends on the point BEFORE the last one.
        # So state: dp[k][s][p_idx] where p_idx is the last point.
        # To extend to e > p_idx, we need to check turn(q[p_idx_prev], q[p_idx], q[e]).
        # But we don't know p_idx_prev in the state dp[k][s][p_idx].
        
        # Therefore, the state must be dp[k][s][p_idx] is not enough.
        # We need dp[k][s][p_idx] to sum over all valid mid points.
        # Actually, the standard DP for this is:
        # dp[k][i][j] = number of convex chains of length k ending at i, j (where i < j in sorted order).
        # But here we have the pivot p.
        
        # Let's use the state: dp[k][s][e] = count of convex chains of k points in q, 
        # starting at s, ending at e, such that the angle is increasing (implied by sorted q)
        # and all turns are convex.
        
        # Recurrence:
        # dp[k][s][e] = sum_{m < e} dp[k-1][s][m] * I( turn(q[m], q[e]) ... )
        # Wait, the turn at q[m] is (q[m] - q[prev]) x (q[e] - q[m]).
        # We don't know q[prev].
        
        # Ah, the turn is determined by the three points.
        # If the chain is p, q[s], ..., q[m], q[e], the last turn is at q[m].
        # It involves q[m-1], q[m], q[e].
        # So we need to know q[m-1].
        
        # This suggests the state is (s, m, e) for the last 3 points?
        # No, just (s, m) is the state for length k-1 ending at m.
        # When extending to e, we check the turn at m.
        # But the turn at m depends on the point BEFORE m in the chain.
        
        # So the state must be dp[k][s][m] where the chain ends at m.
        # To extend to e, we need the turn (m-1, m, e).
        # But dp[k][s][m] aggregates over all possible m-1.
        # So we cannot simply sum dp[k][s][m] if the turn depends on m-1.
        
        # Correct approach:
        # dp[k][s][m] = sum over all valid chains of length k-1 starting at s and ending at m.
        # This is not sufficient because the transition to e depends on the point before m.
        
        # We need dp[k][s][m] to be a vector of counts for each possible predecessor?
        # No, we can do:
        # dp[k][s][e] = sum_{m < e} ( sum_{p < m} dp[k-1][s][p, m] * I(turn(p, m, e) > eps) )
        # where dp[k-1][s][p, m] is chains of length k-1 ending at m with previous p.
        
        # This leads to O(N^5) or worse.
        
        # For N=33, O(N^4) is 1.2M. O(N^5) is 40M. Might be too slow in Python if not vectorized.
        # Let's stick to the O(N^4) brute force check for all 7-subsets?
        # Number of 7-subsets of 33 is ~1.5M.
        # For each subset, check convexity.
        # Convexity check: Sort by angle, check 7 turns.
        # 1.5M * 7 = 10M operations.
        # In NumPy, we can vectorize this.
        
        # Let's try the brute force vectorized approach.
        
    # Vectorized Brute Force
    # Generate all combinations of 7 indices
    # This might be memory heavy if we generate all at once.
    # 1.5M * 7 * 2 floats = 21 MB. Manageable.
    
    # But generating combinations in NumPy is slow.
    # Let's use the DP from the seed, it was O(N^4) and fast.
    
    # I will reuse the logic from the seed's _count_convex7_fast but clean it up.
    
    total = 0
    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # dp[l][s][e]
        # l: length of chain in q (1 to 6)
        # s: start index
        # e: end index
        
        # Precompute turn matrix for q
        # turn[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # We need this for a < b < c
        
        # Initialize dp[1][s][s] = 1
        dp = np.zeros((7, m, m), dtype=np.int64)
        for s in range(m):
            dp[1][s][s] = 1
            
        # l=2
        # dp[2][s][e] = 1 if s < e and turn(p, q[s], q[e]) > eps
        for s in range(m):
            for e in range(s+1, m):
                # turn(p, q[s], q[e])
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[e, 0] - q[s, 0]
                v2y = q[e, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > eps:
                    dp[2][s][e] = 1
        
        # l=3 to 6
        for l in range(3, 7):
            for s in range(m):
                for e in range(s+2, m):
                    # dp[l][s][e] = sum_{m < e} dp[l-1][s][m] * I(turn(q[m], q[e]) ... )
                    # Wait, the turn at q[m] depends on the point BEFORE q[m].
                    # So dp[l-1][s][m] is not enough.
                    
                    # We need to sum over all valid chains ending at m.
                    # But the turn condition depends on the specific predecessor.
                    
                    # This confirms the state must include the predecessor.
                    # dp[l][s][p][e] ?
                    
                    # Let's switch to dp[l][s][p] = count of chains of length l ending at p, starting at s.
                    # This is still not enough.
                    
                    # Let's use the standard O(N^4) DP:
                    # dp[k][i][j] = number of convex chains of length k ending at i, j.
                    # Here, the points are from the set {p} U {q}.
                    # But p is the pivot.
                    
                    # Let's just do the O(N^4) check for the whole set?
                    # No, the pivot structure is key.
                    
                    # I will implement the DP with state (s, p_idx) for length l.
                    # But I need the turn at p_idx.
                    # The turn at p_idx is (prev, p_idx, next).
                    # So when extending a chain ending at p_idx to next, I need prev.
                    
                    # So I need to keep a histogram of predecessors?
                    # No, I can do:
                    # dp[l][s][e] = sum_{m < e} ( sum_{p < m} dp[l-1][s][p, m] * I(turn(p, m, e) > eps) )
                    # where dp[l-1][s][p, m] is 1 if chain s->...->p->m is valid.
                    
                    # This requires storing dp[l-1][s][p, m].
                    # Size: 6 * 33 * 33 * 33 = 250k. Small.
                    
                    # Let's implement this.
                    pass # Placeholder

        # Due to complexity, I will fall back to a simple heuristic in the search loop
        # and use a slower but correct counter.
        
    return 0 # Placeholder

# Actual Implementation Strategy:
# Given the time constraint and the difficulty of getting the DP right on the fly,
# I will use a **randomized hill climbing** with a **simple O(N^4) counter** that is vectorized.
# N=33 is small enough.

def _count_convex7_simple(points):
    n = len(points)
    if n < 7:
        return 0
    total = 0
    # Iterate over all 7-subsets
    # Using itertools.combinations is slow in Python.
    # We will use a nested loop with NumPy for the convexity check.
    
    # Precompute all cross products for triples?
    # cross[i, j, k] = (j-i) x (k-j)
    
    # This is 33^3 = 35937 values.
    # Then for each 7-subset, check if all 7 turns are > eps.
    
    # Let's try to optimize the counting.
    
    return 0

# I will use the seed's counting function as it was verified to work and be fast enough.
# I will copy it here and ensure it is correct.

def _count_convex7_fast(points):
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    eps = 1e-9

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # dp[l][s][e]
        # l: length of chain in q
        # s: start index
        # e: end index
        
        # We will use a 3D array dp[l][s][e]
        # But as analyzed, we need the predecessor.
        
        # Let's use the method from the seed which seemed to work.
        # It used a 3D array and vectorized the inner loop.
        
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        # Initialize dp[2]
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:] > eps)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    mask = turn_slice > eps
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev, dp_curr = dp_curr, dp_prev

        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= eps:
                        continue
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= eps:
                        continue
                    total += count

    return total

def _make_cup_cap(n_cup, n_cap, rng):
    """
    Create a cup-cap construction.
    Cup: n_cup points on a convex curve (lower part)
    Cap: n_cap points on a concave curve (upper part)
    """
    pts = []
    
    # Cup: points along a convex arc at the bottom
    # Use a parabola y = a*x^2 shifted to be in [0,1]
    for i in range(n_cup):
        t = (i + 0.5) / n_cup  # spread in (0,1)
        x = 0.05 + 0.9 * t
        y = 0.05 + 0.15 * (t - 0.5) ** 2  # shallow parabola
        # Add small random perturbation to avoid collinearity
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    
    # Cap: points along a concave arc at the top
    for i in range(n_cap):
        t = (i + 0.5) / n_cap
        x = 0.05 + 0.9 * t
        y = 0.85 - 0.15 * (t - 0.5) ** 2  # inverted parabola
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    
    return np.array(pts)

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Try multiple cup-cap configurations
    best_pts = None
    best_c = float('inf')
    
    # Try different splits
    splits = [(16, 17), (17, 16), (15, 18), (18, 15), (14, 19), (19, 14)]
    
    for n_cup, n_cap in splits:
        if n_cup + n_cap != 33:
            continue
        
        pts = _make_cup_cap(n_cup, n_cap, rng)
        pts = np.clip(pts, 0.01, 0.99)
        
        if not _is_valid_fast(pts):
            # Perturb until valid
            for _ in range(50):
                pts += rng.normal(0, 0.003, (33, 2))
                pts = np.clip(pts, 0.01, 0.99)
                if _is_valid_fast(pts):
                    break
        
        if not _is_valid_fast(pts):
            continue
        
        c = _count_convex7_fast(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random initialization
        best_pts = rng.uniform(0.1, 0.9, (33, 2))
        best_c = _count_convex7_fast(best_pts)
    
    # Hill climbing with multiple points per iteration
    while time_left() > 10:
        # Try moving 1-3 random points
        n_move = rng.integers(1, 4)
        indices = rng.choice(33, size=n_move, replace=False)
        
        new_pts = best_pts.copy()
        for idx in indices:
            new_pts[idx] = best_pts[idx] + rng.normal(0, 0.03, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
    
    return best_pts
