"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Polish + Epigraph."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_DEADLINE = 135.0  # Safety margin before 150s kill


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

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

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

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _soft_min_penalty(violations, beta=50.0):
    """Smooth approximation of sum(max(violation, 0)^2) using soft-min via log-sum-exp."""
    scaled = violations * beta
    max_s = np.max(scaled)
    if max_s > 500:
        return float(np.sum(np.maximum(violations, 0.0) ** 2))
    if max_s < -500:
        return 0.0
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    soft_max = log_sum / beta
    return float(soft_max * soft_max)


def construct():
    global _DEADLINE
    start_time = time.time()
    deadline = start_time + _DEADLINE

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
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

        # Hexagonal packing initial guesses (denser than square grid)
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            rows_needed = int(np.ceil(N / 6)) + 1
            col_spacing = w0 / 6.0
            row_spacing = col_spacing * np.sqrt(3) / 2.0
            idx = 0
            for row in range(rows_needed):
                if idx >= N:
                    break
                y_pos = (row + 0.5) * row_spacing
                if y_pos > h0 - 0.05:
                    break
                offset = (col_spacing / 2.0) if (row % 2 == 1) else 0.0
                for col in range(6):
                    if idx >= N:
                        break
                    x_pos = (col + 0.5) * col_spacing + offset
                    if x_pos < 0.05 or x_pos > w0 - 0.05:
                        continue
                    r0 = min(col_spacing, row_spacing) * 0.48
                    z[3 * idx] = x_pos
                    z[3 * idx + 1] = y_pos
                    z[3 * idx + 2] = r0
                    idx += 1
            if idx < N:
                for i in range(idx, N):
                    z[3 * i] = w0 / 2.0
                    z[3 * i + 1] = h0 / 2.0
                    z[3 * i + 2] = 0.05
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=800)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(1000, max(300, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 150})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.97
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 12:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 8)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        if best_result is not None and time.time() < deadline - 8:
            refined = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 4:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time))
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        if best_result is not None and time.time() < deadline - 2:
            final_refine = _epigraph_refine(best_result, deadline - (time.time() - start_time))
            if final_refine is not None:
                valid, val = check_validity(final_refine)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_refine

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
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
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

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
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
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0008
    max_iter = int(min(10000, time_budget * 400))

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

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 20.0
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
        if max_force > 0.15:
            dt = 0.0004
        elif max_force > 0.05:
            dt = 0.0006
        else:
            dt = 0.0008

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii with binary search for optimal uniform scale
    lo_scale, hi_scale = 1.0, 2.5
    for _ in range(40):
        mid_scale = (lo_scale + hi_scale) / 2.0
        if _is_valid(x, y, r * mid_scale, w, h):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    # Then try per-circle expansion with more iterations
    for _ in range(400):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = min(x[i], w - x[i], y[i], h - y[i])
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
                r[i] = r_max
                improved = True
        if not improved:
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
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Vectorized max-radius computation for speed.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def _max_radii_vectorized(x, y, r, w, h):
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        r_max = r_wall.copy()
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        r_adj = d - r[None, :]
        np.fill_diagonal(r_adj, np.inf)
        r_max = np.minimum(r_max, np.min(r_adj, axis=1))
        return np.maximum(r_max, 0.0)

    def _find_blockers_vectorized(x, y, r, w, h):
        n = len(r)
        blockers = np.full(n, -1, dtype=int)
        min_dists = np.full(n, np.inf)

        d_left = x - r
        d_right = w - x - r
        d_bottom = y - r
        d_top = h - y - r
        wall_dists = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
        wall_min = np.min(wall_dists, axis=1)
        wall_argmin = np.argmin(wall_dists, axis=1)
        min_dists = np.minimum(min_dists, wall_min)
        blockers[wall_min <= min_dists + 1e-12] = wall_argmin[wall_min <= min_dists + 1e-12]

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        gap = d - (r[:, None] + r[None, :])
        np.fill_diagonal(gap, np.inf)
        pair_min = np.min(gap, axis=1)
        pair_argmin = np.argmin(gap, axis=1)
        mask = pair_min < min_dists
        min_dists[mask] = pair_min[mask]
        blockers[mask] = n + pair_argmin[mask]

        return blockers, min_dists

    max_rounds = 100
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        r_max_arr = _max_radii_vectorized(x, y, r, w, h)

        for i in range(n):
            r_max = r_max_arr[i]

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-8:
                    r[i] = new_r
                    improved = True
                    r_max_arr = _max_radii_vectorized(x, y, r, w, h)

            r_max_i = r_max_arr[i]
            if r_max_i < r[i] + 0.003:
                blockers, min_dists = _find_blockers_vectorized(x, y, r, w, h)
                blocker = blockers[i]
                min_dist = min_dists[i]

                if min_dist < 0.01:
                    move_x = 0.0
                    move_y = 0.0
                    if blocker == 0:
                        move_x = 1.0
                    elif blocker == 1:
                        move_x = -1.0
                    elif blocker == 2:
                        move_y = 1.0
                    elif blocker == 3:
                        move_y = -1.0
                    else:
                        j = blocker - n
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-6:
                            move_x = dx / d
                            move_y = dy / d

                    for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002]:
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
                        for step in [0.03, 0.015, 0.008, 0.004]:
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

    lo_scale, hi_scale = 1.0, 2.0
    for _ in range(30):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        if _is_valid(x, y, r * mid_scale, w, h):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _epigraph_refine(z, time_budget):
    """
    Epigraph-based SLSQP refinement: introduce auxiliary variable t >= max constraint violation,
    minimize t while maximizing sum(r). Uses smooth log-sum-exp surrogate for the epigraph constraint.
    """
    z = z.copy().astype(float)
    n = N

    def obj_epigraph(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        # Compute constraint violations (positive = violation)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = r_sum[iu] - d_pairs  # positive if overlapping

        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h
        all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
        all_viol = np.concatenate([overlap, all_viol])

        # Smooth epigraph: t >= max(all_viol) approximated via log-sum-exp
        # Minimize t - alpha * sum(r) + beta * smooth_epigraph_violation
        # where smooth_epigraph_violation = soft_max(all_viol) - t (if positive)
        beta = 100.0
        scaled = all_viol * beta
        max_s = np.max(scaled)
        if max_s > 500:
            soft_max = float(np.max(all_viol))
        elif max_s < -500:
            soft_max = 0.0
        else:
            log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            soft_max = log_sum / beta

        # Penalty for soft_max > 0 (i.e., any violation)
        # Also penalize to drive t towards feasibility
        penalty_weight = 2000.0
        obj = -np.sum(r) + penalty_weight * soft_max * soft_max

        return obj

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
    maxiter = int(min(600, time_budget * 100))
    try:
        res = minimize(obj_epigraph, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        r_shrunk = r * 0.999
        if _is_valid(x, y, r_shrunk, w, h):
            z_opt[2::3][:n] = r_shrunk
            return z_opt
        return None
    except Exception:
        return None


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
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
