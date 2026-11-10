# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Returns (centers, sides) where centers is (10, 2) and sides is (10,).
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin

    def _is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if n != 10:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1.0 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def _max_side_for_square(i, cx, cy, s):
        """
        Compute maximum side for square i given its center and other squares.
        For no overlap with j: need dx >= (s_i+s_j)/2 OR dy >= (s_i+s_j)/2
        So s_i <= max(2*dx - s_j, 2*dy - s_j) for each j.
        Also boundary: s_i <= 2*min(cx_i, 1-cx_i, cy_i, 1-cy_i)
        """
        max_s = min(2.0 * cx[i], 2.0 * (1.0 - cx[i]),
                    2.0 * cy[i], 2.0 * (1.0 - cy[i]))
        max_s = max(max_s, 1e-10)

        for j in range(10):
            if j == i:
                continue
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            limit = max(2.0 * dx - s[j], 2.0 * dy - s[j])
            if limit < max_s:
                max_s = limit

        return max(1e-10, max_s)

    def _compute_optimal_sides(centers, n_iter=100):
        """
        Given centers, compute maximum valid side for each square via fixed-point.
        """
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        s = np.full(10, 0.1)

        for _ in range(n_iter):
            s_new = np.zeros(10)
            for i in range(10):
                s_new[i] = _max_side_for_square(i, cx, cy, s)
            diff = np.max(np.abs(s - s_new))
            s = s_new
            if diff < 1e-14:
                break

        return np.maximum(s, 1e-10)

    def _objective_centers(x):
        """
        Objective: -sum of optimal sides given centers x = [cx0..cx9, cy0..cy9]
        """
        centers = np.column_stack([x[0:10], x[10:20]])
        s = _compute_optimal_sides(centers, n_iter=30)
        return -np.sum(s)

    def _objective_centers_smooth(x, beta=50.0):
        """
        Smooth objective using soft-min for the max operation in Voronoi limits.
        """
        cx = x[0:10]
        cy = x[10:20]
        s = np.full(10, 0.1)

        for _ in range(20):
            s_new = np.zeros(10)
            for i in range(10):
                max_s = min(2.0 * cx[i], 2.0 * (1.0 - cx[i]),
                            2.0 * cy[i], 2.0 * (1.0 - cy[i]))
                max_s = max(max_s, 1e-10)

                for j in range(10):
                    if j == i:
                        continue
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    a = 2.0 * dx - s[j]
                    b = 2.0 * dy - s[j]
                    # soft max of a and b
                    m = max(a, b)
                    smax = m + np.logaddexp(-beta * (a - m), -beta * (b - m)) / beta
                    if smax < max_s:
                        max_s = smax

                s_new[i] = max(1e-10, max_s)

            diff = np.max(np.abs(s - s_new))
            s = s_new
            if diff < 1e-14:
                break

        return -np.sum(s)

    def _repair_and_clamp(centers, sides):
        """
        Ensure validity by shrinking sides if necessary.
        """
        centers = centers.copy()
        sides = sides.copy()

        for i in range(10):
            h = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)

        for _ in range(300):
            overlap_found = False
            for i in range(10):
                for j in range(i + 1, 10):
                    t = (sides[i] + sides[j]) / 2.0
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        overlap_found = True
                        if sides[i] < sides[j]:
                            sides[i] *= 0.85
                        else:
                            sides[j] *= 0.85
                        idx = i if sides[i] < sides[j] else j
                        h = sides[idx] / 2.0
                        centers[idx, 0] = np.clip(centers[idx, 0], h, 1.0 - h)
                        centers[idx, 1] = np.clip(centers[idx, 1], h, 1.0 - h)
            if not overlap_found:
                break

        return centers, sides

    def _optimize_centers(centers, max_time=35.0, method='SLSQP', smooth=False):
        """
        Optimize centers to maximize sum of optimal sides.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1]])
        bounds = [(0.0, 1.0)] * 20

        t0 = time.time()

        def callback(xk):
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")

        obj_fn = _objective_centers_smooth if smooth else _objective_centers

        try:
            if method == 'SLSQP':
                res = minimize(obj_fn, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 400, 'ftol': 1e-12, 'maxfun': 3000},
                               callback=callback)
            else:
                res = minimize(obj_fn, x0, method='COBYLA',
                               options={'maxiter': 400, 'rhobeg': 0.05},
                               callback=callback)

            x_opt = np.clip(res.x, 0.0, 1.0)
            new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
            new_sides = _compute_optimal_sides(new_centers, n_iter=100)

            if _is_valid(new_centers, new_sides):
                return new_centers, new_sides
        except Exception:
            pass

        return None

    def _optimize_full(centers, sides, max_time=35.0, method='SLSQP'):
        """
        Full 30-dim optimization with penalty for overlaps.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1], sides])
        bounds = [(0.0, 1.0)] * 20 + [(0.01, 1.0)] * 10

        t0 = time.time()

        def _obj(x, penalty_weight=500.0):
            cx = x[0:10]
            cy = x[10:20]
            s = x[20:30]

            obj = -np.sum(s)

            for i in range(10):
                for v in [cx[i] - s[i] / 2.0, (1.0 - s[i] / 2.0) - cx[i],
                          cy[i] - s[i] / 2.0, (1.0 - s[i] / 2.0) - cy[i]]:
                    if v < 0:
                        obj += penalty_weight * v * v

            eps = 1e-8
            for i in range(10):
                for j in range(i + 1, 10):
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    t = (s[i] + s[j]) / 2.0
                    ov_x = t - dx
                    ov_y = t - dy
                    smax_u = 0.5 * (ov_x + np.sqrt(ov_x ** 2 + eps))
                    smax_v = 0.5 * (ov_y + np.sqrt(ov_y ** 2 + eps))
                    obj += penalty_weight * smax_u * smax_v

            return obj

        def callback(xk):
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")

        try:
            if method == 'SLSQP':
                res = minimize(_obj, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 250, 'ftol': 1e-10, 'maxfun': 2000},
                               callback=callback)
            else:
                res = minimize(_obj, x0, method='COBYLA',
                               options={'maxiter': 250, 'rhobeg': 0.05},
                               callback=callback)

            x_opt = res.x
            new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
            new_sides = x_opt[20:30]
            new_centers, new_sides = _repair_and_clamp(new_centers, new_sides)

            if _is_valid(new_centers, new_sides):
                return new_centers, new_sides
        except Exception:
            pass

        return None

    def _polish(centers, sides, max_time=20.0):
        """
        High-precision polish: alternate between optimizing centers (with exact sides)
        and fine-tuning sides via local search.
        """
        c = centers.copy()
        s = sides.copy()
        t0 = time.time()

        for iteration in range(3):
            if time.time() - t0 > max_time:
                break

            # Optimize centers
            result = _optimize_centers(c, max_time=max_time - (time.time() - t0),
                                       method='SLSQP', smooth=False)
            if result is not None:
                c, s = result

            # Fine-tune: try small perturbations of sides
            rng = np.random.default_rng(int(time.time() * 1000) % 10000 + iteration)
            for _ in range(5):
                if time.time() - t0 > max_time:
                    break
                s_trial = s * (1 + rng.normal(0, 0.02, size=10))
                c_trial, s_trial = _repair_and_clamp(c, s_trial)
                if _is_valid(c_trial, s_trial) and np.sum(s_trial) > np.sum(s):
                    c, s = c_trial, s_trial

        return c, s

    # --- Main Search Loop ---

    best_centers = None
    best_sides = None
    best_score = -1.0

    def _update_best(c, s):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(c, s):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = c.copy()
                best_sides = s.copy()

    def init_grid_33():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        coords = [0.15, 0.5, 0.85]
        idx = 0
        for r in range(3):
            for col in range(3):
                c[idx, 0] = coords[col]
                c[idx, 1] = coords[r]
                s[idx] = 0.3
                idx += 1
        c[9, 0] = 0.5
        c[9, 1] = 0.5
        s[9] = 0.05
        return c, s

    def init_corner():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        c[0] = [0.2, 0.2]; s[0] = 0.4
        c[1] = [0.8, 0.2]; s[1] = 0.4
        c[2] = [0.2, 0.8]; s[2] = 0.4
        c[3] = [0.8, 0.8]; s[3] = 0.4
        c[4] = [0.5, 0.1]; s[4] = 0.2
        c[5] = [0.1, 0.5]; s[5] = 0.2
        c[6] = [0.9, 0.5]; s[6] = 0.2
        c[7] = [0.5, 0.9]; s[7] = 0.2
        c[8] = [0.35, 0.35]; s[8] = 0.1
        c[9] = [0.65, 0.65]; s[9] = 0.1
        return c, s

    def init_stripe():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.18
        return c, s

    def init_random(seed):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.1, 0.9, size=(10, 2))
        s = rng.uniform(0.1, 0.4, size=10)
        return c, s

    def init_diagonal():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            t = i / 9.0
            c[i, 0] = 0.15 + 0.7 * t
            c[i, 1] = 0.15 + 0.7 * t
            s[i] = 0.15
        return c, s

    inits = [
        init_grid_33,
        init_corner,
        init_stripe,
        init_diagonal,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        lambda: init_random(777),
        lambda: init_random(2024),
        lambda: init_random(555),
        lambda: init_random(314),
        lambda: init_random(271),
    ]

    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break

        c, s = init_fn()
        c, s = _repair_and_clamp(c, s)
        _update_best(c, s)

        # Center optimization (exact)
        if time.time() < deadline - 80:
            result_v = _optimize_centers(c, max_time=30.0, method='SLSQP', smooth=False)
            if result_v is not None:
                _update_best(result_v[0], result_v[1])

        # Center optimization (smooth)
        if time.time() < deadline - 60:
            result_v2 = _optimize_centers(c, max_time=25.0, method='SLSQP', smooth=True)
            if result_v2 is not None:
                _update_best(result_v2[0], result_v2[1])

        # Full optimization
        if time.time() < deadline - 50:
            x0 = np.concatenate([c[:, 0], c[:, 1], s])
            result = _optimize_full(c, s, max_time=25.0, method='SLSQP')
            if result is not None:
                _update_best(result[0], result[1])

        # COBYLA refinement from best
        if best_centers is not None and time.time() < deadline - 40:
            x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
            result2 = _optimize_full(best_centers, best_sides, max_time=20.0, method='COBYLA')
            if result2 is not None:
                _update_best(result2[0], result2[1])

        # Perturbation-based local search
        if best_centers is not None and time.time() < deadline - 30:
            rng = np.random.default_rng(i * 100 + 1)
            c3 = best_centers + rng.normal(0, 0.03, size=(10, 2))
            s3 = best_sides * (1 + rng.normal(0, 0.08, size=10))
            c3, s3 = _repair_and_clamp(c3, s3)
            _update_best(c3, s3)

            if time.time() < deadline - 25:
                result3 = _optimize_centers(c3, max_time=20.0, method='SLSQP', smooth=False)
                if result3 is not None:
                    _update_best(result3[0], result3[1])

    # Final polish
    if best_centers is not None and time.time() < deadline - 20:
        c_pol, s_pol = _polish(best_centers, best_sides, max_time=15.0)
        _update_best(c_pol, s_pol)

    # Fallback
    if best_centers is None:
        best_centers = np.zeros((10, 2))
        best_sides = np.zeros(10)
        for i in range(10):
            r = i // 5
            col = i % 5
            best_centers[i, 0] = 0.1 + col * 0.2
            best_centers[i, 1] = 0.1 + r * 0.2
            best_sides[i] = 0.1
        best_centers, best_sides = _repair_and_clamp(best_centers, best_sides)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
