"""Circle packing in perimeter-4 rectangle: SLSQP joint optimization + multi-round expansion + local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
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


def _smooth_max(a, b, k=50.0):
    """Smooth approximation of max(a,b) using log-sum-exp."""
    m = np.maximum(a, b)
    return m + np.logaddexp(k * (a - m), k * (b - m)) / k


def _objective(z, penalty_weight=8000.0):
    """Objective for SLSQP with smooth penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty (smooth)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty (smooth)
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))

    # Aspect ratio penalty
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _force_relax(x, y, r, w, h, max_iter=800, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
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

        # Adaptive step
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


def _greedy_expand_multi_round(x, y, r, w, h, time_budget, max_rounds=30):
    """Multi-round greedy expansion: iterate until no further improvement."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
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
    return x, y, r


def _local_search_positions(x, y, r, w, h, time_budget, rng):
    """Try small random perturbations of positions to find better packings."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for attempt in range(8000):
        if time.time() > time_budget:
            break
        # Pick a random circle and nudge it
        i = rng.integers(0, N)
        scale = 0.003 + 0.004 * (attempt % 5) / 5.0  # gradually increase perturbation
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        nx = x[i] + dx
        ny = y[i] + dy

        # Check validity of new position
        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
            continue
        valid = True
        for j in range(N):
            if i == j:
                continue
            ddx = nx - x[j]
            ddy = ny - y[j]
            if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                valid = False
                break
        if valid:
            x[i] = nx
            y[i] = ny
            # Try to expand all circles
            x2, y2, r2 = _greedy_expand_multi_round(x, y, r, w, h, time_budget, max_rounds=15)
            new_sum = np.sum(r2)
            if new_sum > best_sum + 1e-10:
                best_sum = new_sum
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                x, y, r = x2, y2, r2
            else:
                # Revert
                x[i] -= dx
                y[i] -= dy
                r = best_r.copy()

    return best_x, best_y, best_r


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()

    max_iter = int(min(3000, (time_budget - time.time()) * 120))
    if max_iter < 150:
        max_iter = 150
    dt = 0.0008
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
        iu = np.triu_indices(n, k=1)
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

    # Binary search for max uniform scale factor
    lo, hi = 1.0, 1.5
    for _ in range(40):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid

    r = r * lo

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization of positions and radii."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.3, 1.7)]
    maxiter = int(min(1000, (time_budget - time.time()) * 80))
    if maxiter < 80:
        maxiter = 80

    try:
        res = minimize(_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            # Try to repair
            r_repair = r_opt * 0.98
            if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass

    return None


def _generate_grid_layout(w0, h0, pattern='square'):
    """Generate initial layout with different patterns."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    if pattern == 'square':
        rows = 4
        cols = 6  # 4*6=24, use first 21
        for i in range(N):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) * (w0 / cols)
            y[i] = (row + 0.5) * (h0 / rows)
            r[i] = min(w0 / cols, h0 / rows) * 0.45

    elif pattern == 'staggered':
        rows = 5
        cols = 5
        for i in range(N):
            row = i // cols
            col = i % cols
            offset = 0.5 * (w0 / cols) if row % 2 == 1 else 0.0
            x[i] = (col + 0.5) * (w0 / cols) + offset
            y[i] = (row + 0.5) * (h0 / rows)
            r[i] = min(w0 / cols, h0 / rows) * 0.42
            # Clip to stay in bounds
            x[i] = np.clip(x[i], r[i], w0 - r[i])

    elif pattern == 'hex':
        # Hexagonal packing approximation
        row_h = 0.866  # sqrt(3)/2
        rows = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            spacing_x = w0 / 5.5
            spacing_y = h0 / 4.5
            offset = spacing_x / 2 if row % 2 == 1 else 0.0
            x[i] = (col + 0.5) * spacing_x + offset
            y[i] = (row + 0.5) * spacing_y
            r[i] = min(spacing_x, spacing_y) * 0.40
            x[i] = np.clip(x[i], r[i], w0 - r[i])
            y[i] = np.clip(y[i], r[i], h0 - r[i])

    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    rng = np.random.default_rng(42)

    # Diverse initial layouts with various aspect ratios and patterns
    configs = []
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5, 0.9, 1.1, 1.3, 0.65]
    patterns = ['square', 'staggered', 'hex']

    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
        for pat in patterns:
            configs.append((w0, h0, pat))

    for w0, h0, pat in configs:
        if time.time() > deadline - 25:
            break

        x, y, r = _generate_grid_layout(w0, h0, pat)

        # Force relaxation
        x, y = _force_relax(x, y, r, w0, h0, max_iter=600)

        if not _is_valid(x, y, r, w0, h0):
            r_shrink = r * 0.92
            if not _is_valid(x, y, r_shrink, w0, h0):
                continue
            r = r_shrink

        # Multi-round greedy expansion
        x_e, y_e, r_e = _greedy_expand_multi_round(x, y, r, w0, h0, deadline - 20)
        if _is_valid(x_e, y_e, r_e, w0, h0):
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e.copy(), y_e.copy(), r_e.copy(), w0, h0)

        # Polish and grow
        polished = _polish_positions(x_e, y_e, r_e, w0, h0, deadline - 20)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w0, h0)

        # SLSQP optimization if time permits
        if time.time() < deadline - 35 and best_result is not None:
            x_b, y_b, r_b, w_b, h_b = best_result
            opt_result = _slsqp_optimize(x_b, y_b, r_b, w_b, h_b, deadline - 30)
            if opt_result is not None:
                x_o, y_o, r_o, w_o, h_o = opt_result
                val = float(np.sum(r_o))
                if val > best_value:
                    best_value = val
                    best_result = (x_o.copy(), y_o.copy(), r_o.copy(), w_o, h_o)

                # Multi-round expansion after SLSQP
                if time.time() < deadline - 20:
                    x_ex, y_ex, r_ex = _greedy_expand_multi_round(x_o, y_o, r_o, w_o, h_o, deadline - 15)
                    if _is_valid(x_ex, y_ex, r_ex, w_o, h_o):
                        val_ex = float(np.sum(r_ex))
                        if val_ex > best_value:
                            best_value = val_ex
                            best_result = (x_ex.copy(), y_ex.copy(), r_ex.copy(), w_o, h_o)

    # Local search on best result
    if best_result is not None and time.time() < deadline - 15:
        x_b, y_b, r_b, w_b, h_b = best_result
        x_ls, y_ls, r_ls = _local_search_positions(x_b, y_b, r_b, w_b, h_b, deadline - 10, rng)
        if _is_valid(x_ls, y_ls, r_ls, w_b, h_b):
            val_ls = float(np.sum(r_ls))
            if val_ls > best_value:
                best_value = val_ls
                best_result = (x_ls.copy(), y_ls.copy(), r_ls.copy(), w_b, h_b)

        # Final polish
        if time.time() < deadline - 5:
            polished_final = _polish_positions(x_ls, y_ls, r_ls, w_b, h_b, deadline - 3)
            if polished_final is not None:
                x_pf, y_pf, r_pf = polished_final
                val_pf = float(np.sum(r_pf))
                if val_pf > best_value:
                    best_value = val_pf
                    best_result = (x_pf.copy(), y_pf.copy(), r_pf.copy(), w_b, h_b)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity
    for _ in range(20):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
