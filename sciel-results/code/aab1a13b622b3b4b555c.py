# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    Points are sorted lexicographically, for each pivot the later points are
    sorted by atan2 angle around it, and a turn p -> q -> r counts as convex
    only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

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

        # Precompute turn(a, b, c) = (q[b]-q[a]) x (q[c]-q[b]) for a < b < c
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # dp[s][p_idx][c_idx] = # convex chains starting at s, prev p_idx, ending c_idx
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        # dp[2]
        for s in range(m):
            valid_c = np.where(turn_p[s, s + 1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s + 1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx + 1:]
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    mask = turn_slice > 1e-9
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx + 1:] = contrib
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
                    if cross1 <= 1e-9:
                        continue
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= 1e-9:
                        continue
                    total += count

    return total

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

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

def _generate_initial_points(rng):
    """
    Generate points in 6 tight clusters arranged in a hexagonal pattern.
    This structure tends to minimize convex 7-gons by keeping points locally dense.
    """
    pts = []
    # 6 clusters arranged in a hexagon
    centers = [
        (0.5, 0.15), (0.85, 0.35), (0.85, 0.65),
        (0.5, 0.85), (0.15, 0.65), (0.15, 0.35)
    ]
    # Distribute 33 points: 6 clusters of 5 + 1 center
    for cx, cy in centers:
        for _ in range(5):
            x = cx + rng.normal(0, 0.015)
            y = cy + rng.normal(0, 0.015)
            pts.append((x, y))
    
    # 1 point near center
    x = rng.uniform(0.45, 0.55)
    y = rng.uniform(0.45, 0.55)
    pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 200:
        pts += rng.normal(0, 0.002, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _anneal(pts, rng, time_limit, k_clusters=6):
    """
    Simulated annealing with a strong pull toward cluster centroids.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    cur_pts = best_pts.copy()
    cur_c = best_c
    start_time = time.time()

    # 6 hexagonal centers + 1 center
    centers = np.array([
        [0.5, 0.15], [0.85, 0.35], [0.85, 0.65],
        [0.5, 0.85], [0.15, 0.65], [0.15, 0.35], [0.5, 0.5]
    ])

    T0 = 0.10
    Tend = 0.003

    while time.time() - start_time < time_limit:
        frac = min(1.0, (time.time() - start_time) / time_limit)
        T = T0 * (Tend / T0) ** frac
        pull = 0.30 + 0.5 * frac

        d2 = ((cur_pts[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        target = centers[assign]
        step = cur_pts + pull * (target - cur_pts) + rng.normal(0, T, (33, 2))
        step = np.clip(step, 0.01, 0.99)

        if not _is_valid_fast(step):
            for _ in range(30):
                step += rng.normal(0, 0.0015, (33, 2))
                step = np.clip(step, 0.01, 0.99)
                if _is_valid_fast(step):
                    break
            if not _is_valid_fast(step):
                continue

        c = _count_convex7_fast(step)
        if c < cur_c or rng.random() < np.exp(-(c - cur_c) / max(T, 1e-6)):
            cur_pts = step
            cur_c = c
            if c < best_c:
                best_c = c
                best_pts = step.copy()
                if best_c <= 1:
                    break
        else:
            pass

    return best_pts, best_c

def _hill_climb(pts, rng, time_limit):
    """
    Fine-grained hill climbing with multi-point moves to polish the annealed configuration.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()

    no_improve = 0
    max_no_improve = 120
    step_size = 0.04

    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False

        if no_improve > 30 and rng.random() < 0.5:
            n_move = int(rng.integers(2, 5))
        else:
            n_move = 1

        idx = rng.choice(33, size=n_move, replace=False)
        new_pts = best_pts.copy()
        for i in idx:
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)

        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size *= 0.9
            elif c == best_c and best_c > 1:
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if not improved and no_improve % 25 == 0:
            step_size = min(0.15, step_size * 1.3)

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    best_pts = _generate_initial_points(rng)
    best_c = _count_convex7_fast(best_pts)
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c

    # Phase 1: annealing to collapse into clusters
    while time_left() > 120:
        chunk = min(60, time_left() - 120)
        if chunk < 20:
            break
        best_pts, best_c = _anneal(best_pts, rng, chunk)
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
        if best_ever_c <= 1:
            break

    # Phase 2: hill climb to polish
    while time_left() > 30:
        chunk = min(30, time_left() - 30)
        if chunk < 10:
            break
        best_pts, best_c = _hill_climb(best_ever_pts, rng, chunk)
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
        if best_ever_c <= 1:
            break

    return best_ever_pts
# EVOLVE-BLOCK-END
