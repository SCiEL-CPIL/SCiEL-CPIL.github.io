"""Circle packing in perimeter-4 rectangle: direct construction + SLSQP + force polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_DEADLINE = 145.0  # safety margin before 150s kill


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _objective(z, penalty_weight=5000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2
    return obj


def _smooth_objective(z, penalty_weight=5000.0, beta=50.0):
    """Smooth log-sum-exp penalty for a smoother landscape."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    # Smooth max via log-sum-exp: max(a,0) ≈ (1/beta)*log(1+exp(beta*a))
    raw_overlap = r_sum[iu] - d_pairs
    smooth_overlap = (1.0 / beta) * np.logaddexp(0.0, beta * raw_overlap)
    obj += penalty_weight * np.sum(smooth_overlap * smooth_overlap)
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2
    return obj


def _force_relax(x, y, r, w, h, max_iter=800, dt_init=0.003):
    n = N
    dt = dt_init
    iu = np.triu_indices(n, k=1)
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        if np.max(overlap) < 1e-9:
            break
        f_pairs = overlap * 12.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)
        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    return x, y


def _polish_positions(x, y, r, w, h, time_budget):
    n = N
    x = x.copy()
    y = y.copy()
    iu = np.triu_indices(n, k=1)
    max_iter = int(min(3000, time_budget * 150))
    dt = 0.001
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]
        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # High-precision radial expansion via binary search
    if _is_valid(x, y, r, w, h):
        lo, hi = 1.0, 1.5  # search up to 50% expansion
        # First find an upper bound
        while hi < 2.0 and _is_valid(x, y, r * hi, w, h):
            lo = hi
            hi *= 1.5
        # Binary search for exact maximum
        for _ in range(60):
            if time.time() > time_budget:
                break
            mid = (lo + hi) / 2
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        r = r * lo
    return x, y, r


def _expand_individual(x, y, r, w, h, time_budget):
    n = N
    improved = False
    for i in range(n):
        if time.time() > time_budget:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-5:
            r[i] = r_max
            improved = True
    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None


def _move_circle_to_better_position(i, x, y, r, w, h, time_budget):
    if time.time() > time_budget:
        return x, y, r, False
    best_r = r[i]
    best_x, best_y = x[i], y[i]

    def _max_radius_at(xi, yi):
        r_wall = min(xi, w - xi, yi, h - yi)
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = xi - x[j]
            dy = yi - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        return max(0.0, r_max)

    step = 0.02
    for dx in np.arange(-0.1, 0.11, step):
        for dy in np.arange(-0.1, 0.11, step):
            if time.time() > time_budget:
                break
            xi = x[i] + dx
            yi = y[i] + dy
            if xi < r[i] or xi > w - r[i] or yi < r[i] or yi > h - r[i]:
                continue
            r_trial = _max_radius_at(xi, yi)
            if r_trial > best_r + 1e-6:
                best_r = r_trial
                best_x, best_y = xi, yi
    if best_r > r[i] + 1e-5:
        x[i] = best_x
        y[i] = best_y
        r[i] = best_r
        return x, y, r, True
    return x, y, r, False


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    # Phase 1: diverse initial constructions
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5, 1.1, 0.9, 1.3]

    for w0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        best_grid = None
        best_grid_r = 0
        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < N or rows * cols > N + 6:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_grid = min(cell_w, cell_h) * 0.48
                if r_grid > best_grid_r:
                    best_grid_r = r_grid
                    best_grid = (rows, cols)

        if best_grid is None:
            continue

        rows, cols = best_grid
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        for i in range(N):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) * (w0 / cols)
            y[i] = (row + 0.5) * (h0 / rows)
            r[i] = min(w0 / cols, h0 / rows) * 0.45

        x, y = _force_relax(x, y, r, w0, h0, max_iter=500)

        if not _is_valid(x, y, r, w0, h0):
            r *= 0.95
            if not _is_valid(x, y, r, w0, h0):
                r *= 0.95
                if not _is_valid(x, y, r, w0, h0):
                    continue

        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # Phase 2: SLSQP optimization with multiple perturbations
    if best_result is not None and time.time() < deadline - 35:
        x_best, y_best, r_best, w_best, h_best = best_result
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x_best
        z[1::3][:N] = y_best
        z[2::3][:N] = r_best
        z[-1] = w_best

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(150, remaining * 30)))

        for attempt in range(4):
            if time.time() > deadline - 35:
                break
            if attempt == 0:
                z_try = z.copy()
                obj_fn = _objective
            elif attempt == 1:
                z_try = z.copy()
                obj_fn = _smooth_objective
            else:
                z_try = z.copy()
                z_try[:2*N] += np.random.normal(0, 0.01, size=2*N)
                z_try[2*N:3*N] *= (1 + np.random.normal(0, 0.02, size=N))
                z_try[2*N:3*N] = np.maximum(z_try[2*N:3*N], 0.01)
                z_try[-1] = np.clip(z_try[-1], 0.3, 1.7)
                obj_fn = _objective if attempt % 2 == 0 else _smooth_objective

            try:
                res = minimize(obj_fn, z_try, method='SLSQP',
                               bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
                z_opt = res.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt

                if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                    val = float(np.sum(r_opt))
                    if val > best_value:
                        best_value = val
                        best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
                else:
                    for scale in [0.98, 0.95, 0.9]:
                        r_repair = r_opt * scale
                        if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                            val = float(np.sum(r_repair))
                            if val > best_value:
                                best_value = val
                                best_result = (x_opt.copy(), y_opt.copy(), r_repair.copy(), w_opt, h_opt)
                            break
            except Exception:
                pass

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 15:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 3b: Re-run SLSQP on polished result for further refinement
    if best_result is not None and time.time() < deadline - 20:
        x_best, y_best, r_best, w_best, h_best = best_result
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x_best
        z[1::3][:N] = y_best
        z[2::3][:N] = r_best
        z[-1] = w_best
        try:
            res = minimize(_smooth_objective, z, method='SLSQP',
                           bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                           options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 60})
            z_opt = res.x
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt
            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                val = float(np.sum(r_opt))
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
        except Exception:
            pass

    # Phase 4: Individual circle optimization
    if best_result is not None and time.time() < deadline - 8:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 3
        x_curr, y_curr, r_curr = x_best.copy(), y_best.copy(), r_best.copy()
        for i in range(N):
            if time.time() > remaining + start_time:
                break
            x_curr, y_curr, r_curr, improved = _move_circle_to_better_position(
                i, x_curr, y_curr, r_curr, w_best, h_best, remaining + start_time)
        if _is_valid(x_curr, y_curr, r_curr, w_best, h_best):
            val = float(np.sum(r_curr))
            if val > best_value:
                best_value = val
                best_result = (x_curr, y_curr, r_curr, w_best, h_best)

    # Phase 5: Final expansion
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 1
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    # Phase 5b: Final high-precision binary search on scale
    if best_result is not None and time.time() < deadline - 1:
        x_best, y_best, r_best, w_best, h_best = best_result
        lo, hi = 1.0, 1.3
        while hi < 2.0 and _is_valid(x_best, y_best, r_best * hi, w_best, h_best):
            lo = hi
            hi *= 1.4
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
            mid = (lo + hi) / 2
            if _is_valid(x_best, y_best, r_best * mid, w_best, h_best):
                lo = mid
            else:
                hi = mid
        r_final = r_best * lo
        val_final = float(np.sum(r_final))
        if val_final > best_value:
            best_value = val_final
            best_result = (x_best, y_best, r_final, w_best, h_best)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Tiny shrink for numerical safety
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
