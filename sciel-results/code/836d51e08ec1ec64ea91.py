import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 20.0


def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming with vectorized closing checks.
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

        # Precompute turn matrix
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # Precompute turn from pivot p
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # DP
        dp_prev = np.zeros((m, m, m), dtype=np.int64)

        for s in range(m):
            valid_c = np.where(turn_p[s, s + 1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s + 1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    vec = dp_prev[s, s:p_idx, p_idx]
                    if np.any(vec > 0):
                        mask = (turn_mat[s:p_idx, p_idx, p_idx + 1:] > 1e-9).astype(np.int64)
                        contrib = vec @ mask
                        dp_curr[s, p_idx, p_idx + 1:] = contrib
            dp_prev = dp_curr

        # Vectorized closing turn checks
        valid_mask = dp_prev > 0
        if not np.any(valid_mask):
            continue

        s_idx, p_idx_arr, c_idx_arr = np.where(valid_mask)
        counts = dp_prev[s_idx, p_idx_arr, c_idx_arr]

        # Vectorized cross1: (q[c]-q[p]) x (p-q[c])
        qc = q[c_idx_arr]
        qp = q[p_idx_arr]
        v1 = qc - qp
        v2 = p[None, :] - qc
        cross1 = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

        # Vectorized cross2: (p-q[c]) x (q[s]-p)
        qs = q[s_idx]
        v3 = p[None, :] - qc
        v4 = qs - p[None, :]
        cross2 = v3[:, 0] * v4[:, 1] - v3[:, 1] * v4[:, 0]

        valid_both = (cross1 > 1e-9) & (cross2 > 1e-9)
        total += np.sum(counts[valid_both])

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


def _make_points_from_1d(xs, ys):
    """Build 33x2 array from 1D x and y arrays."""
    return np.column_stack((xs, ys))


def _is_valid_1d(xs, ys):
    """Fast validity check for 1D representation (monotone x, monotone y)."""
    n = len(xs)
    # Check bounds
    if np.any(xs < -1e-6) or np.any(xs > 1 + 1e-6):
        return False
    if np.any(ys < -1e-6) or np.any(ys > 1 + 1e-6):
        return False
    # Check distinctness: since x is monotone, just check x differences
    dx = np.diff(xs)
    if np.any(dx < 1e-6):
        return False
    dy = np.diff(ys)
    if np.any(dy < 1e-6):
        return False
    # Check no three collinear
    # For monotone x, three points i<j<k are collinear iff
    # (xs[j]-xs[i])*(ys[k]-ys[i]) == (xs[k]-xs[i])*(ys[j]-ys[i])
    # We need |cross| >= 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (xs[j] - xs[i]) * (ys[k] - ys[i]) - (xs[k] - xs[i]) * (ys[j] - ys[i])
                if abs(cross) < 1e-9:
                    return False
    return True


def _generate_initial_1d(rng):
    """
    Generate initial 1D representation: strictly increasing x in [0.05, 0.95],
    and y as a concave function of x with small noise.
    This structurally minimizes convex 7-gons because the points lie on a
    concave curve, so most 7-subsets have reflex vertices.
    """
    xs = np.linspace(0.05, 0.95, 33)
    # Concave curve: y = 0.5 * (1 - (2x-1)^2) + small noise
    base_y = 0.5 * (1 - (2 * xs - 1)**2)
    noise = rng.normal(0, 0.003, 33)
    ys = base_y + noise

    # Ensure strict monotonicity
    for i in range(1, 33):
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1e-4

    # Clip to [0.01, 0.99]
    xs = np.clip(xs, 0.01, 0.99)
    ys = np.clip(ys, 0.01, 0.99)

    # Re-ensure monotonicity after clipping
    for i in range(1, 33):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-4
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1e-4

    # Final clip
    xs = np.clip(xs, 0.01, 0.99)
    ys = np.clip(ys, 0.01, 0.99)

    attempts = 0
    while not _is_valid_1d(xs, ys) and attempts < 200:
        # Perturb y values
        idx = rng.integers(0, 33)
        ys[idx] += rng.normal(0, 0.002)
        ys = np.clip(ys, 0.01, 0.99)
        # Re-ensure monotonicity
        for i in range(1, 33):
            if ys[i] <= ys[i - 1]:
                ys[i] = ys[i - 1] + 1e-4
        ys = np.clip(ys, 0.01, 0.99)
        attempts += 1

    return xs, ys


def _hill_climb_1d(xs, ys, rng, time_limit):
    """
    Hill climb in 1D space: perturb either x or y coordinates while maintaining
    monotonicity. This reduces the search space and makes each evaluation faster
    since the structure is more constrained.
    """
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_c = _count_convex7_fast(_make_points_from_1d(best_xs, best_ys))
    start_time = time.time()

    no_improve = 0
    max_no_improve = 100
    step_size = 0.03

    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False

        # Choose which coordinate to perturb: x or y
        if rng.random() < 0.5:
            # Perturb x
            i = rng.integers(0, 33)
            new_xs = best_xs.copy()
            new_ys = best_ys.copy()
            new_xs[i] = best_xs[i] + rng.normal(0, step_size)
            new_xs[i] = np.clip(new_xs[i], 0.01, 0.99)

            # Re-ensure monotonicity of x
            if i > 0 and new_xs[i] <= new_xs[i - 1]:
                new_xs[i] = new_xs[i - 1] + 1e-4
            if i < 32 and new_xs[i + 1] <= new_xs[i]:
                new_xs[i + 1] = new_xs[i] + 1e-4
            new_xs = np.clip(new_xs, 0.01, 0.99)
        else:
            # Perturb y
            i = rng.integers(0, 33)
            new_xs = best_xs.copy()
            new_ys = best_ys.copy()
            new_ys[i] = best_ys[i] + rng.normal(0, step_size)
            new_ys[i] = np.clip(new_ys[i], 0.01, 0.99)

            # Re-ensure monotonicity of y
            if i > 0 and new_ys[i] <= new_ys[i - 1]:
                new_ys[i] = new_ys[i - 1] + 1e-4
            if i < 32 and new_ys[i + 1] <= new_ys[i]:
                new_ys[i + 1] = new_ys[i] + 1e-4
            new_ys = np.clip(new_ys, 0.01, 0.99)

        if _is_valid_1d(new_xs, new_ys):
            pts = _make_points_from_1d(new_xs, new_ys)
            c = _count_convex7_fast(pts)
            if c < best_c:
                best_c = c
                best_xs = new_xs
                best_ys = new_ys
                improved = True
                no_improve = 0
                step_size *= 0.95
            elif c == best_c and best_c > 1:
                if rng.random() < 0.3:
                    best_xs = new_xs
                    best_ys = new_ys
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if not improved and no_improve % 10 == 0:
            # Kick: perturb multiple coordinates
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_xs = best_xs.copy()
            new_ys = best_ys.copy()
            for i in idx:
                new_xs[i] = best_xs[i] + rng.normal(0, step_size * 2)
                new_ys[i] = best_ys[i] + rng.normal(0, step_size * 2)
                new_xs[i] = np.clip(new_xs[i], 0.01, 0.99)
                new_ys[i] = np.clip(new_ys[i], 0.01, 0.99)

            # Re-ensure monotonicity
            for i in range(1, 33):
                if new_xs[i] <= new_xs[i - 1]:
                    new_xs[i] = new_xs[i - 1] + 1e-4
                if new_ys[i] <= new_ys[i - 1]:
                    new_ys[i] = new_ys[i - 1] + 1e-4
            new_xs = np.clip(new_xs, 0.01, 0.99)
            new_ys = np.clip(new_ys, 0.01, 0.99)

            if _is_valid_1d(new_xs, new_ys):
                pts = _make_points_from_1d(new_xs, new_ys)
                c = _count_convex7_fast(pts)
                if c < best_c:
                    best_c = c
                    best_xs = new_xs
                    best_ys = new_ys
                    no_improve = 0
                    step_size *= 0.95

        if no_improve % 50 == 0:
            step_size = min(0.15, step_size * 1.5)

    return best_xs, best_ys, best_c


def _polish_1d(xs, ys, rng, time_limit):
    """Final high-precision polish pass with smaller step sizes in 1D space."""
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_c = _count_convex7_fast(_make_points_from_1d(best_xs, best_ys))
    start_time = time.time()

    step_size = 0.005
    no_improve = 0
    max_no_improve = 200

    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        if rng.random() < 0.5:
            i = rng.integers(0, 33)
            new_xs = best_xs.copy()
            new_ys = best_ys.copy()
            new_xs[i] = best_xs[i] + rng.normal(0, step_size)
            new_xs[i] = np.clip(new_xs[i], 0.01, 0.99)
            if i > 0 and new_xs[i] <= new_xs[i - 1]:
                new_xs[i] = new_xs[i - 1] + 1e-4
            if i < 32 and new_xs[i + 1] <= new_xs[i]:
                new_xs[i + 1] = new_xs[i] + 1e-4
            new_xs = np.clip(new_xs, 0.01, 0.99)
        else:
            i = rng.integers(0, 33)
            new_xs = best_xs.copy()
            new_ys = best_ys.copy()
            new_ys[i] = best_ys[i] + rng.normal(0, step_size)
            new_ys[i] = np.clip(new_ys[i], 0.01, 0.99)
            if i > 0 and new_ys[i] <= new_ys[i - 1]:
                new_ys[i] = new_ys[i - 1] + 1e-4
            if i < 32 and new_ys[i + 1] <= new_ys[i]:
                new_ys[i + 1] = new_ys[i] + 1e-4
            new_ys = np.clip(new_ys, 0.01, 0.99)

        if _is_valid_1d(new_xs, new_ys):
            pts = _make_points_from_1d(new_xs, new_ys)
            c = _count_convex7_fast(pts)
            if c < best_c:
                best_c = c
                best_xs = new_xs
                best_ys = new_ys
                no_improve = 0
                step_size *= 0.9
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve % 30 == 0:
            step_size = max(0.001, step_size * 1.3)

    return best_xs, best_ys, best_c


def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    best_xs, best_ys = _generate_initial_1d(rng)
    best_pts = _make_points_from_1d(best_xs, best_ys)
    best_c = _count_convex7_fast(best_pts)

    best_ever_xs = best_xs.copy()
    best_ever_ys = best_ys.copy()
    best_ever_c = best_c

    print(f"Initial C: {best_c}")

    while time_left() > 90:
        chunk_time = min(45, time_left() - 90)
        if chunk_time < 10:
            break

        best_xs, best_ys, best_c = _hill_climb_1d(best_xs, best_ys, rng, chunk_time)

        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_xs = best_xs.copy()
            best_ever_ys = best_ys.copy()

        print(f"After hill climb: C={best_c}, best_ever={best_ever_c}, t={time.time() - start_time:.1f}")

        if best_ever_c <= 1:
            break

        # Restart
        if time_left() > 120:
            new_xs, new_ys = _generate_initial_1d(rng)
            new_pts = _make_points_from_1d(new_xs, new_ys)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                best_xs = new_xs
                best_ys = new_ys
                best_c = new_c
                best_ever_xs = new_xs.copy()
                best_ever_ys = new_ys.copy()
                best_ever_c = new_c
                print(f"Restart improved to C: {best_c}")
            else:
                best_xs = best_ever_xs.copy()
                best_ys = best_ever_ys.copy()
                best_c = best_ever_c

    # Final polish pass
    if time_left() > 30:
        polish_time = min(30, time_left() - 20)
        if polish_time > 10:
            best_xs, best_ys, best_c = _polish_1d(best_ever_xs, best_ever_ys, rng, polish_time)
            if best_c < best_ever_c:
                best_ever_c = best_c
                best_ever_xs = best_xs.copy()
                best_ever_ys = best_ys.copy()
                print(f"After polish: C={best_c}, t={time.time() - start_time:.1f}")

    result = _make_points_from_1d(best_ever_xs, best_ever_ys)
    return result
