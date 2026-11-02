"""Circle packing in perimeter-4 rectangle: SLSQP + Delaunay-guided expansion + Advanced Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
_NUM_PAIRS = N * (N - 1) // 2
_IU = np.triu_indices(N, k=1)


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    d_sq_pairs = d_sq[_IU]
    r_sum_pairs = r_sum[_IU]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _smooth_objective(z, beta=50.0):
    """Smooth log-sum-exp based objective."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    d_sq_pairs = d_sq[_IU]
    r_sum_pairs = r_sum[_IU]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    overlap_vals = r_sum_pairs - d_pairs

    shifted = beta * overlap_vals
    shifted = np.clip(shifted, -500, 500)
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_max_overlap = lse / beta

    hard_overlap = np.maximum(overlap_vals, 0.0)
    quad_penalty = 100.0 * np.sum(hard_overlap * hard_overlap)

    obj += 500.0 * smooth_max_overlap + quad_penalty

    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    shifted_b = beta * all_viol
    shifted_b = np.clip(shifted_b, -500, 500)
    max_b = np.max(shifted_b)
    lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
    smooth_max_viol = lse_b / beta

    hard_viol = np.maximum(all_viol, 0.0)
    obj += 500.0 * smooth_max_viol + 100.0 * np.sum(hard_viol * hard_viol)

    if w < 0.5:
        obj += 200.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 200.0 * (w - 1.5) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal guess
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # Staggered grid
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        sx = w0 / (cols + 0.3)
        sy = h0 / (rows + 0.3)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
            cy = (row + 0.5) * sy
            r0 = min(sx, sy) * 0.49
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # Corner-biased
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        corner_positions = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for k, (cx, cy) in enumerate(corner_positions):
            z[3 * k] = cx * w0
            z[3 * k + 1] = cy * h0
            z[3 * k + 2] = 0.12
        remaining = N - 4
        r_side = int(np.ceil(np.sqrt(remaining)))
        cx0, cy0 = 0.3, 0.3
        cw, ch = 0.4, 0.4
        for k in range(remaining):
            row, col = divmod(k, r_side)
            cx = (cx0 + (col + 0.5) / r_side * cw) * w0
            cy = (cy0 + (row + 0.5) / r_side * ch) * h0
            r0 = min(cw / r_side, ch / r_side) * 0.45
            idx = 4 + k
            z[3 * idx] = cx
            z[3 * idx + 1] = cy
            z[3 * idx + 2] = r0
        z[-1] = w0
        guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            pass

        if time.time() < deadline - 25:
            try:
                res2 = minimize(_smooth_objective, z0_relaxed, method='SLSQP',
                                bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 60})
                z_opt2 = res2.x
                valid2, val2 = check_validity(z_opt2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = z_opt2.copy()
            except Exception:
                pass

    # --- Phase 2: Advanced Polish ---
    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # NEW: Delaunay-guided expansion
        if best_result is not None and time.time() < deadline - 12:
            delaunay_exp = _delaunay_guided_expansion(best_result, deadline - (time.time() - start_time) - 8)
            if delaunay_exp is not None:
                valid, val = check_validity(delaunay_exp)
                if valid and val > best_value:
                    best_value = val
                    best_result = delaunay_exp

        # Multi-start perturbation + expansion
        if best_result is not None and time.time() < deadline - 12:
            rng2 = np.random.default_rng(123)
            for trial in range(5):
                if time.time() > deadline - 12:
                    break
                z_pert = best_result.copy()
                x_p = z_pert[0::3][:N] + rng2.normal(0, 0.01, N)
                y_p = z_pert[1::3][:N] + rng2.normal(0, 0.01, N)
                w_p = z_pert[-1]
                h_p = 2.0 - w_p
                r_p = z_pert[2::3][:N]
                x_p = np.clip(x_p, r_p, w_p - r_p)
                y_p = np.clip(y_p, r_p, h_p - r_p)
                z_pert[0::3][:N] = x_p
                z_pert[1::3][:N] = y_p

                z_pert = _force_relax_unified(z_pert, max_iter=200)

                expanded2 = _expansion_relaxation_polish(z_pert, deadline - (time.time() - start_time) - 10)
                if expanded2 is not None:
                    valid2, val2 = check_validity(expanded2)
                    if valid2 and val2 > best_value:
                        best_value = val2
                        best_result = expanded2

        # Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # Final high-precision polish
        if best_result is not None and time.time() < deadline - 2:
            polished2 = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[_IU]
            dy_pairs = dy[_IU]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            fx_contrib = np.zeros(n)
            fy_contrib = np.zeros(n)
            np.add.at(fx_contrib, _IU[0], ux * f_pairs)
            np.add.at(fx_contrib, _IU[1], -ux * f_pairs)
            np.add.at(fy_contrib, _IU[0], uy * f_pairs)
            np.add.at(fy_contrib, _IU[1], -uy * f_pairs)
            fx += fx_contrib
            fy += fy_contrib

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
            dx_pairs = dx[_IU]
            dy_pairs = dy[_IU]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            fx_contrib = np.zeros(n)
            fy_contrib = np.zeros(n)
            np.add.at(fx_contrib, _IU[0], ux * f_pairs)
            np.add.at(fx_contrib, _IU[1], -ux * f_pairs)
            np.add.at(fy_contrib, _IU[0], uy * f_pairs)
            np.add.at(fy_contrib, _IU[1], -uy * f_pairs)
            fx += fx_contrib
            fy += fy_contrib

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """Iteratively expand circles and shift positions to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            dx_i = x[i] - x
            dy_i = y[i] - y
            d_i = np.sqrt(dx_i * dx_i + dy_i * dy_i)
            d_i[i] = np.inf
            r_allowed = d_i - r
            r_max = min(r_wall, np.min(r_allowed))
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True

            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]

            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'

            dx_i = x[i] - x
            dy_i = y[i] - y
            d_i = np.sqrt(dx_i * dx_i + dy_i * dy_i)
            d_i[i] = np.inf
            gaps = d_i - (r[i] + r)
            min_gap = np.min(gaps)
            if min_gap < min_dist:
                min_dist = min_gap
                blocker = int(np.argmin(gaps))

            if min_dist < 0.02:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d

                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                else:
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    grow_factor = 1.001
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _delaunay_guided_expansion(z, time_budget):
    """
    NEW: Use Delaunay triangulation to identify the contact graph,
    then iteratively expand the most constrained circles and shift
    their neighbors to maintain feasibility. This exploits the
    geometric structure of the packing to find better local optima.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Build Delaunay triangulation to identify contact graph
    def _get_contact_graph(x, y, w, h):
        """Get contact graph from Delaunay triangulation + boundary points."""
        # Add boundary points
        margin = 0.02
        boundary_pts = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
            [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
            [w/4, 0], [3*w/4, 0], [w, h/4], [w, 3*h/4],
            [3*w/4, h], [w/4, h], [0, 3*h/4], [0, h/4],
        ])
        all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
        n_total = len(all_pts)

        try:
            tri = Delaunay(all_pts)
        except Exception:
            return np.zeros((n, n), dtype=bool)

        contact = np.zeros((n, n), dtype=bool)
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    ia, ib = simplex[a], simplex[b]
                    if ia < n and ib < n:
                        # Check if they're actually in contact (close)
                        dx = x[ia] - x[ib]
                        dy = y[ia] - y[ib]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < r[ia] + r[ib] + 0.05:
                            contact[ia, ib] = True
                            contact[ib, ia] = True
        return contact

    # Iterative expansion guided by contact graph
    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        contact = _get_contact_graph(x, y, w, h)

        # Find the circle with the most room to expand (least constrained)
        # but also prioritize circles that are blocking others
        expansion_priority = np.zeros(n)
        for i in range(n):
            # Room to expand
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            dx_i = x[i] - x
            dy_i = y[i] - y
            d_i = np.sqrt(dx_i * dx_i + dy_i * dy_i)
            d_i[i] = np.inf
            r_allowed = d_i - r
            r_max = min(r_wall, np.min(r_allowed))
            room = max(0.0, r_max - r[i])

            # How many contacts
            n_contacts = int(np.sum(contact[i]))

            # Prioritize: more room AND fewer contacts (easier to expand)
            # But also: if a circle is blocking many others, expanding it helps
            expansion_priority[i] = room * (1.0 + 0.1 * n_contacts)

        # Try to expand each circle in priority order
        improved = False
        order = np.argsort(-expansion_priority)
        for i in order:
            if time.time() > time_budget:
                break

            # Calculate max radius
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            dx_i = x[i] - x
            dy_i = y[i] - y
            d_i = np.sqrt(dx_i * dx_i + dy_i * dy_i)
            d_i[i] = np.inf
            r_allowed = d_i - r
            r_max = min(r_wall, np.min(r_allowed))
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-9):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

                    # After expanding, try to shift neighbors to make room
                    # for further expansion
                    for j in range(n):
                        if j == i:
                            continue
                        # Check if j is now close to i
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < r[i] + r[j] + 0.03:
                            # Push j away
                            if d > 1e-6:
                                push_x = (dx / d) * 0.005
                                push_y = (dy / d) * 0.005
                            else:
                                push_x = 0.005
                                push_y = 0.0
                            new_xj = x[j] + push_x
                            new_yj = y[j] + push_y
                            if _is_valid_move(x, y, r, w, h, j, new_xj, new_yj):
                                x[j] = new_xj
                                y[j] = new_yj

        if not improved:
            break

    # Final uniform expansion
    grow_factor = 1.0005
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h, tol=1e-9):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h, tol=1e-9):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

    dx = new_x - x
    dy = new_y - y
    d2 = dx * dx + dy * dy
    d2[i] = np.inf
    r_sum = r[i] + r
    if np.any(d2 < r_sum * r_sum - 1e-9):
        return False
    return True


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - z_opt[-1]

        if _is_valid(x, y, r, w, h):
            return z_opt

        r_shrunk = r * 0.999
        if _is_valid(x, y, r_shrunk, w, h):
            z_opt[2::3][:n] = r_shrunk
            return z_opt
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    d_sq_pairs = d_sq[_IU]
    r_sum_pairs = r_sum[_IU]
    if np.any(d_sq_pairs < r_sum_pairs ** 2 - tol):
        return False
    return True


def _high_precision_polish(z, time_budget):
    """High-precision polish: iterative radius growth with position adjustment."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Fine-tune positions
    dt = 0.0002
    max_iter = int(min(5000, time_budget * 200))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 25.0
            dx_pairs = dx[_IU]
            dy_pairs = dy[_IU]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            fx_contrib = np.zeros(n)
            fy_contrib = np.zeros(n)
            np.add.at(fx_contrib, _IU[0], ux * f_pairs)
            np.add.at(fx_contrib, _IU[1], -ux * f_pairs)
            np.add.at(fy_contrib, _IU[0], uy * f_pairs)
            np.add.at(fy_contrib, _IU[1], -uy * f_pairs)
            fx += fx_contrib
            fy += fy_contrib

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Phase 2: Alternating individual expansion + micro position adjustment
    for round_num in range(80):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-9:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True

            if improved:
                dx_move = 0.0
                dy_move = 0.0
                d_left = x[i] - r[i]
                d_right = w - x[i] - r[i]
                d_bot = y[i] - r[i]
                d_top = h - y[i] - r[i]
                min_d = min(d_left, d_right, d_bot, d_top)
                if d_left == min_d:
                    dx_move = 0.003
                elif d_right == min_d:
                    dx_move = -0.003
                elif d_bot == min_d:
                    dy_move = 0.003
                elif d_top == min_d:
                    dy_move = -0.003

                if dx_move != 0 or dy_move != 0:
                    new_xi = x[i] + dx_move
                    new_yi = y[i] + dy_move
                    if _is_valid_move(x, y, r, w, h, i, new_xi, new_yi):
                        x[i] = new_xi
                        y[i] = new_yi

    # Phase 3: Uniform expansion with binary search
    lo_factor = 1.0
    hi_factor = 1.02
    for _ in range(40):
        if time.time() > time_budget:
            break
        mid_factor = (lo_factor + hi_factor) / 2.0
        r_trial = r * mid_factor
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            lo_factor = mid_factor
            r = r_trial
        else:
            hi_factor = mid_factor

    # Phase 4: Final micro-expansion
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * 1.00005
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None
