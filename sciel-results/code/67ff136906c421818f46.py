import time
import numpy as np

_DEADLINE = 625.0
_SAFE_MARGIN = 5.0

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

    # Collinearity check - vectorized
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            # Cross product of (p_j - p_i) with (p_k - p_i) for all k
            v1 = pts[j] - pts[i]
            v2 = pts - pts[i]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            # Exclude i and j themselves
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(np.abs(cross[mask]) < 1e-9):
                return False
    return True

def _count_convex7_fast(points):
    """
    Count convex 7-gons using the DP method from the task description.
    Points sorted lexicographically, for each pivot, later points sorted by angle.
    """
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

        # Precompute turn matrix: turn[a,b,c] = (q[b]-q[a]) x (q[c]-q[b])
        # Only need a < b < c
        # turn_mat[a, b, c] for a < b < c
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1 = q[b] - q[a]
                v2 = q[b+1:] - q[b]  # shape (m-b-1, 2)
                cross_vals = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                turn_mat[a, b, b+1:] = cross_vals

        # turn_p[s, c] = turn(p, q[s], q[c]) = (q[s]-p) x (q[c]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1 = q[s] - p
            v2 = q[s+1:] - q[s]  # shape (m-s-1, 2)
            cross_vals = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            turn_p[s, s+1:] = cross_vals

        # DP: dp[chain_len][s][e] = number of convex chains of length chain_len
        # starting at q[s], ending at q[e], with all intermediate indices between s and e
        # Chain: p -> q[s] -> ... -> q[e]
        # dp[2][s][e] = 1 if turn(p, q[s], q[e]) > eps
        # dp[l][s][e] = sum over prev in (s, e) of dp[l-1][s][prev] * [turn(q[prev], q[e-1], q[e]) > eps]
        # Wait, the chain is p -> q[s] -> q[i1] -> ... -> q[e]
        # The turns are:
        # - turn(p, q[s], q[i1]) > eps
        # - turn(q[s], q[i1], q[i2]) > eps
        # - ...
        # - turn(q[prev], q[e], p) > eps (closing)
        # - turn(q[e], p, q[s]) > eps (closing)
        
        # Let's define: dp[l][s][e] = number of convex chains of l points from q[s] to q[e]
        # (including s and e) such that all consecutive turns are convex (left turns)
        # and the chain can be extended to form a convex polygon with p.
        
        # Actually, let's follow the standard approach:
        # For a fixed pivot p, we want to count subsets {q[i1], ..., q[i6]} with i1 < i2 < ... < i6
        # such that p, q[i1], ..., q[i6] form a convex 7-gon.
        
        # The DP: dp[k][i][j] = number of convex chains of length k ending at q[j], 
        # starting from q[i] (where i is the first point after p), with all intermediate points between i and j.
        
        # Let's use: dp[l][s][e] where l is the number of points in the chain (including s and e),
        # s < e, and the chain is q[s] -> ... -> q[e] with all indices in [s, e].
        
        # Base: dp[2][s][e] = 1 if turn(p, q[s], q[e]) > eps
        # Recurrence: dp[l][s][e] = sum_{prev in (s, e)} dp[l-1][s][prev] * [turn(q[prev], q[e-1], q[e]) > eps]
        # Wait, that's not right. The turn at q[e] should be turn(q[prev], q[e], next) but we don't have next yet.
        
        # Let me reconsider. The standard DP for counting convex k-gons with a fixed starting point:
        # For pivot p, sort other points by angle.
        # dp[i][j][k] = number of convex chains of length k from i to j (i < j)
        # where the chain starts at q[i] and ends at q[j], and all points are in [i, j].
        # dp[i][j][2] = 1 if turn(p, q[i], q[j]) > eps
        # dp[i][j][k] = sum_{m in (i, j)} dp[i][m][k-1] * [turn(q[m], q[j-1], q[j]) > eps]
        # No, that's still not quite right.
        
        # Let's use a cleaner formulation:
        # Let f(l, s, e) = number of convex chains of l points (l >= 2) starting at q[s] and ending at q[e],
        # where all intermediate points have indices strictly between s and e, and all turns are left turns.
        # The chain is: q[s] -> q[i1] -> ... -> q[e] where s < i1 < ... < e.
        # Turns: turn(q[s], q[i1], q[i2]) > eps, ..., turn(q[prev], q[e-1], q[e]) > eps
        # Note: the turn at q[s] with p is handled separately.
        
        # f(2, s, e) = 1 if turn(p, q[s], q[e]) > eps
        # f(l, s, e) = sum_{prev in (s, e)} f(l-1, s, prev) * [turn(q[prev], q[e-1], q[e]) > eps]
        # Wait, the turn at the second-to-last point prev to last point e is turn(q[prev], q[e], next) but we don't have next.
        
        # I think the correct recurrence is:
        # f(l, s, e) = sum_{prev in (s, e)} f(l-1, s, prev) * [turn(q[prev], q[e], ?) > eps]
        # But we need the turn at q[e] to be with the next point, which we don't have.
        
        # Let me look at this differently. The convex chain p -> q[s] -> q[i1] -> ... -> q[e] 
        # has these turns:
        # 1. turn(p, q[s], q[i1]) > eps
        # 2. turn(q[s], q[i1], q[i2]) > eps
        # ...
        # k. turn(q[i_{k-2}], q[i_{k-1}], q[e]) > eps
        # Then we need to close: turn(q[e], p, q[s]) > eps and turn(q[i_{k-1}], q[e], p) > eps
        
        # So the DP should count chains where all internal turns are convex,
        # and then we check the two closing turns.
        
        # Let's define: g(l, s, e) = number of chains q[s] -> ... -> q[e] of l points (s and e included)
        # where all consecutive turns within the chain are left turns.
        # g(2, s, e) = 1 (no internal turns to check for 2 points)
        # g(l, s, e) = sum_{prev in (s, e)} g(l-1, s, prev) * [turn(q[prev], q[e-1], q[e]) > eps]
        # Wait, for l=3: g(3, s, e) = sum_{prev in (s,e)} g(2, s, prev) * [turn(q[prev], q[prev], q[e]) > eps]
        # That doesn't make sense.
        
        # Let me use a different indexing. Let the chain be q[i0] -> q[i1] -> ... -> q[i_{l-1}] where s = i0 < i1 < ... < i_{l-1} = e.
        # The turns are: turn(q[i0], q[i1], q[i2]), turn(q[i1], q[i2], q[i3]), ..., turn(q[i_{l-3}], q[i_{l-2}], q[i_{l-1}])
        # So there are l-2 turns.
        
        # g(l, s, e) = number of such chains with all l-2 turns being left turns.
        # g(2, s, e) = 1
        # g(3, s, e) = 1 if turn(q[s], q[prev], q[e]) > eps for some prev in (s,e), but we need to sum over prev.
        # Actually: g(3, s, e) = sum_{prev in (s,e)} [turn(q[s], q[prev], q[e]) > eps]
        # g(l, s, e) = sum_{prev in (s,e)} g(l-1, s, prev) * [turn(q[prev], q[e-1], q[e]) > eps]
        # No, that's wrong. The turn at prev to e is turn(q[prev], q[e], next) but we don't have next.
        
        # I think the issue is that the turn at the second-to-last point involves the last point and the next point (which is p for the closing).
        
        # Let me just implement it directly for l=7 (6 points after p):
        # We need to count subsets {i1, i2, i3, i4, i5, i6} with s = i1 < i2 < i3 < i4 < i5 < i6 = e
        # such that:
        # - turn(p, q[s], q[i2]) > eps
        # - turn(q[s], q[i2], q[i3]) > eps
        # - turn(q[i2], q[i3], q[i4]) > eps
        # - turn(q[i3], q[i4], q[i5]) > eps
        # - turn(q[i4], q[i5], q[e]) > eps
        # - turn(q[i5], q[e], p) > eps
        # - turn(q[e], p, q[s]) > eps
        
        # This is O(m^6) which is too slow. We need DP.
        
        # Let's use: dp[k][s][e] = number of convex chains of k points from q[s] to q[e] (s < e)
        # where the chain is q[s] -> ... -> q[e] and all turns within the chain (excluding the turn at s with p and the closing turns) are left turns.
        # dp[2][s][e] = 1 if turn(p, q[s], q[e]) > eps
        # dp[k][s][e] = sum_{prev in (s,e)} dp[k-1][s][prev] * [turn(q[prev], q[e-1], q[e]) > eps]
        # Wait, for k=3: dp[3][s][e] = sum_{prev in (s,e)} dp[2][s][prev] * [turn(q[prev], q[prev], q[e]) > eps]
        # That's still wrong.
        
        # I'll use a different approach. Let's define:
        # dp[k][i][j] = number of convex chains of length k (k points) starting at q[i] and ending at q[j],
        # where i < j, and all points in the chain have indices in [i, j], and all consecutive turns are left turns.
        # The chain is q[i] -> q[i1] -> ... -> q[j] where i < i1 < ... < j.
        # Number of turns: k-2
        # dp[2][i][j] = 1
        # dp[k][i][j] = sum_{m in (i,j)} dp[k-1][i][m] * [turn(q[m], q[j-1], q[j]) > eps]
        # No, the turn at m to j is turn(q[m], q[j], next) but we don't have next.
        
        # I think the correct recurrence is:
        # dp[k][i][j] = sum_{m in (i,j)} dp[k-1][i][m] * [turn(q[m], q[j-1], q[j]) > eps]
        # But this assumes the second-to-last point is j-1, which is not true.
        
        # Let me just do it with explicit loops for 6 points since m is at most 32.
        # For each pivot p, m <= 32. We need to count 6-element subsets from m points.
        # C(32, 6) = 906,192 which is feasible.
        
        count = 0
        for s in range(m):
            # Check turn(p, q[s], q[?]) - we'll check as we go
            for i2 in range(s + 1, m):
                # turn(p, q[s], q[i2])
                v1 = q[s] - p
                v2 = q[i2] - q[s]
                if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                    continue
                for i3 in range(i2 + 1, m):
                    # turn(q[s], q[i2], q[i3])
                    v1 = q[i2] - q[s]
                    v2 = q[i3] - q[i2]
                    if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                        continue
                    for i4 in range(i3 + 1, m):
                        # turn(q[i2], q[i3], q[i4])
                        v1 = q[i3] - q[i2]
                        v2 = q[i4] - q[i3]
                        if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                            continue
                        for i5 in range(i4 + 1, m):
                            # turn(q[i3], q[i4], q[i5])
                            v1 = q[i4] - q[i3]
                            v2 = q[i5] - q[i4]
                            if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                                continue
                            for i6 in range(i5 + 1, m):
                                # turn(q[i4], q[i5], q[i6])
                                v1 = q[i5] - q[i4]
                                v2 = q[i6] - q[i5]
                                if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                                    continue
                                # Check closing turns
                                # turn(q[i5], q[i6], p)
                                v1 = q[i6] - q[i5]
                                v2 = p - q[i6]
                                if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                                    continue
                                # turn(q[i6], p, q[s])
                                v1 = p - q[i6]
                                v2 = q[s] - p
                                if v1[0] * v2[1] - v1[1] * v2[0] <= eps:
                                    continue
                                count += 1
        total += count

    return total

def _make_layered(n_layers, points_per_layer, rng):
    """
    Create a layered point configuration.
    Points are arranged in horizontal layers, each nearly collinear.
    This structure minimizes convex 7-gons because any 7 points spanning
    multiple layers will have interior points.
    """
    pts = []
    y_positions = np.linspace(0.1, 0.9, n_layers)
    
    for layer_idx, y_base in enumerate(y_positions):
        n_pts = points_per_layer[layer_idx]
        # Spread points horizontally with small vertical perturbation
        x_positions = np.linspace(0.05, 0.95, n_pts)
        for x in x_positions:
            # Small vertical perturbation to avoid collinearity
            y = y_base + rng.normal(0, 0.003)
            x_perturbed = x + rng.normal(0, 0.002)
            pts.append((x_perturbed, y))
    
    return np.array(pts)

def _make_convex_with_interior(n_convex, n_interior, rng):
    """
    Place n_convex points on a convex curve (e.g., circle) and n_interior points
    inside the convex hull. This ensures that any subset containing interior
    points is not in convex position.
    """
    pts = []
    
    # Convex points on a circle (slightly perturbed)
    angles = np.linspace(0, 2 * np.pi, n_convex, endpoint=False)
    for angle in angles:
        r = 0.45 + rng.normal(0, 0.01)
        x = 0.5 + r * np.cos(angle)
        y = 0.5 + r * np.sin(angle)
        pts.append((x, y))
    
    # Interior points
    for _ in range(n_interior):
        x = rng.uniform(0.2, 0.8)
        y = rng.uniform(0.2, 0.8)
        pts.append((x, y))
    
    return np.array(pts)

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate multiple initial configurations
    candidates = []
    
    # Layered configurations
    for n_layers in [3, 4, 5, 6]:
        base_per_layer = 33 // n_layers
        remainder = 33 % n_layers
        points_per_layer = [base_per_layer + (1 if i < remainder else 0) for i in range(n_layers)]
        pts = _make_layered(n_layers, points_per_layer, rng)
        pts = np.clip(pts, 0.01, 0.99)
        if _is_valid_fast(pts):
            candidates.append(pts)
    
    # Convex + interior configurations
    for n_convex in [20, 24, 28, 30]:
        n_interior = 33 - n_convex
        pts = _make_convex_with_interior(n_convex, n_interior, rng)
        pts = np.clip(pts, 0.01, 0.99)
        if _is_valid_fast(pts):
            candidates.append(pts)
    
    # Random fallback
    candidates.append(rng.uniform(0.1, 0.9, (33, 2)))
    
    # Evaluate all candidates
    best_pts = None
    best_c = float('inf')
    
    for pts in candidates:
        if not _is_valid_fast(pts):
            continue
        c = _count_convex7_fast(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = rng.uniform(0.1, 0.9, (33, 2))
        best_c = _count_convex7_fast(best_pts)
    
    # Hill climbing with adaptive step size
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 200
    
    while time_left() > 15 and no_improve_count < max_no_improve:
        # Try moving 1-2 random points
        n_move = rng.integers(1, 3)
        indices = rng.choice(33, size=n_move, replace=False)
        
        new_pts = best_pts.copy()
        for idx in indices:
            new_pts[idx] = best_pts[idx] + rng.normal(0, step_size, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_pts = new_pts
                best_c = c
                no_improve_count = 0
                # Slightly increase step size on improvement
                step_size = min(0.1, step_size * 1.1)
            else:
                no_improve_count += 1
                # Decrease step size when stuck
                if no_improve_count % 50 == 0:
                    step_size *= 0.9
        else:
            no_improve_count += 1
    
    return best_pts
