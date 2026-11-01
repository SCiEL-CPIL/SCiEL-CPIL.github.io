import numpy as np
import time
from scipy.optimize import minimize, fsolve

N = 21


def _vectorized_objective(z, r, w, penalty_weight=500.0):
    """Vectorized smooth objective for position optimization."""
    x = z[:N]
    y = z[N:2*N]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return 1e8

    # Wall violations
    x_low = np.maximum(r - x, 0.0)
    x_high = np.maximum(x - (w - r), 0.0)
    y_low = np.maximum(r - y, 0.0)
    y_high = np.maximum(y - (h - r), 0.0)
    wall_pen = penalty_weight * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                                 np.sum(y_low ** 2) + np.sum(y_high ** 2))

    # Pairwise overlaps (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
    r_sum = r[iu[0]] + r[iu[1]]
    overlap = np.maximum(r_sum - d_pairs, 0.0)
    overlap_pen = penalty_weight * np.sum(overlap ** 2)

    return wall_pen + overlap_pen


def _force_relax_unified(x, y, r, w, max_iter=800, tol=1e-10):
    """Vectorized force-directed relaxation for circle packing."""
    h = 2.0 - w
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    dt = 0.003
    min_dt = 0.00005

    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces (vectorized)
        x_low_viol = np.maximum(r - x, 0.0)
        x_high_viol = np.maximum(x - (w - r), 0.0)
        y_low_viol = np.maximum(r - y, 0.0)
        y_high_viol = np.maximum(y - (h - r), 0.0)
        fx -= x_low_viol * 200
        fx += x_high_viol * 200
        fy -= y_low_viol * 200
        fy += y_high_viol * 200

        # Circle-circle forces (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-24))
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(r_sum - d, 0.0)

        # Force magnitude per pair
        f_mag = overlap * 30.0
        # Unit vectors (avoid division by zero)
        d_safe = np.maximum(d, 1e-12)
        ux = dx / d_safe
        uy = dy / d_safe

        # Sum forces: for each i, sum over j != i of f_mag[i,j] * ux[i,j]
        # Only upper triangle contributes (symmetric)
        iu = np.triu_indices(N, k=1)
        fx_i = np.zeros(N)
        fy_i = np.zeros(N)
        np.add.at(fx_i, iu[0], f_mag[iu] * ux[iu])
        np.add.at(fy_i, iu[0], f_mag[iu] * uy[iu])
        np.add.at(fx_i, iu[1], -f_mag[iu] * ux[iu])
        np.add.at(fy_i, iu[1], -f_mag[iu] * uy[iu])

        fx += fx_i
        fy += fy_i

        # Track max overlap for convergence
        max_overlap = max(np.max(overlap), np.max(x_low_viol), np.max(x_high_viol),
                         np.max(y_low_viol), np.max(y_high_viol))
        if max_overlap < tol:
            break

        force_mag = np.max(np.sqrt(fx * fx + fy * fy))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.05:
            dt = min(0.008, dt * 1.15)

        x += dt * fx
        y += dt * fy

        # Clip to bounds
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _is_valid(x, y, r, w, tol=1e-8):
    """Check validity of circle packing configuration."""
    h = 2.0 - w
    if w <= tol or h <= tol:
        return False
    if np.any(r <= tol):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    iu = np.triu_indices(N, k=1)
    r_sum = r[iu[0]] + r[iu[1]]
    if np.any(d_sq[iu] < r_sum ** 2 - tol):
        return False
    return True


def _is_valid_equal_r(x, y, r, w, tol=1e-8):
    """Check validity for equal-radius configuration."""
    h = 2.0 - w
    if w <= tol or h <= tol or r <= tol:
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < (2 * r) ** 2 - tol):
        return False
    return True


def _constraint_refine(x, y, r, w, max_iter=200, tol=1e-13):
    """Newton-Raphson refinement for exact contact configuration."""
    h = 2.0 - w
    x = x.copy().astype(float)
    y = y.copy().astype(float)

    def find_active_constraints(x, y, eps=1e-7):
        """Identify active constraints (contacts)."""
        constraints = []
        for i in range(N):
            if abs(x[i] - r[i]) < eps:
                constraints.append(('wall', i, 0, 'left'))
            if abs(x[i] - (w - r[i])) < eps:
                constraints.append(('wall', i, 0, 'right'))
            if abs(y[i] - r[i]) < eps:
                constraints.append(('wall', i, 0, 'bottom'))
            if abs(y[i] - (h - r[i])) < eps:
                constraints.append(('wall', i, 0, 'top'))
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if abs(d - (r[i] + r[j])) < eps:
                    constraints.append(('circle', i, j, ''))
        return constraints

    def compute_residual(x, y, cons):
        res = np.zeros(len(cons))
        for k, (ctype, i, j, side) in enumerate(cons):
            if ctype == 'wall':
                if side == 'left':
                    res[k] = x[i] - r[i]
                elif side == 'right':
                    res[k] = x[i] - (w - r[i])
                elif side == 'bottom':
                    res[k] = y[i] - r[i]
                elif side == 'top':
                    res[k] = y[i] - (h - r[i])
            else:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-15
                res[k] = d - (r[i] + r[j])
        return res

    def compute_jacobian(x, y, cons):
        n_cons = len(cons)
        J = np.zeros((n_cons, 2 * N))
        for k, (ctype, i, j, side) in enumerate(cons):
            if ctype == 'wall':
                if side == 'left' or side == 'right':
                    J[k, i] = 1.0
                else:
                    J[k, N + i] = 1.0
            else:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-15
                J[k, i] = dx / d
                J[k, N + i] = dy / d
                J[k, j] = -dx / d
                J[k, N + j] = -dy / d
        return J

    for it in range(max_iter):
        cons = find_active_constraints(x, y)
        if len(cons) == 0:
            break

        res = compute_residual(x, y, cons)
        if np.max(np.abs(res)) < tol:
            break

        J = compute_jacobian(x, y, cons)
        try:
            delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            break

        # Damped Newton step
        alpha = 1.0
        for _ in range(25):
            x_trial = x + alpha * delta[:N]
            y_trial = y + alpha * delta[N:]

            valid = True
            if np.any(x_trial < r - 1e-10) or np.any(x_trial > w - r + 1e-10):
                valid = False
            elif np.any(y_trial < r - 1e-10) or np.any(y_trial > h - r + 1e-10):
                valid = False
            else:
                dx = x_trial[:, None] - x_trial[None, :]
                dy = y_trial[:, None] - y_trial[None, :]
                d_sq = dx * dx + dy * dy
                iu = np.triu_indices(N, k=1)
                r_sum = r[iu[0]] + r[iu[1]]
                if np.any(d_sq[iu] < r_sum ** 2 - 1e-10):
                    valid = False

            if valid:
                x = x_trial
                y = y_trial
                break
            alpha *= 0.5

        if alpha < 1e-10:
            break

    return x, y


def _expansion_relaxation_polish(x, y, r, w, max_rounds=30, time_budget=20.0):
    """Iteratively grow individual circle radii with relaxation."""
    h = 2.0 - w
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()

    for round_idx in range(max_rounds):
        if time.time() - start > time_budget:
            break
        improved = False
        # Try growing each circle
        for i in range(N):
            if time.time() - start > time_budget:
                break
            # Try growing circle i
            for attempt in range(30):
                r_trial = r.copy()
                r_trial[i] *= 1.002
                if r_trial[i] > min(w, h) / 2.0 - 0.001:
                    break

                # Relax positions
                x_new, y_new = _force_relax_unified(x, y, r_trial, w, max_iter=200, tol=1e-10)
                if _is_valid(x_new, y_new, r_trial, w):
                    x, y, r = x_new, y_new, r_trial
                    improved = True
                else:
                    break

        if not improved:
            break

    return x, y, r


def _smooth_epigraph_polish(x, y, r, w, time_budget=15.0):
    """Smooth epigraph-style refinement using auxiliary variable."""
    h = 2.0 - w
    start = time.time()

    def epigraph_obj(z):
        xz = z[:N]
        yz = z[N:2*N]
        rz = z[2*N:3*N]
        wv = z[3*N]
        hv = 2.0 - wv

        if wv < 0.3 or hv < 0.3:
            return 1e8

        # Objective: minimize sum of radii (negated for maximization)
        obj = -np.sum(rz)

        # Wall violations
        x_low = np.maximum(rz - xz, 0.0)
        x_high = np.maximum(xz - (wv - rz), 0.0)
        y_low = np.maximum(rz - yz, 0.0)
        y_high = np.maximum(yz - (hv - rz), 0.0)
        wall_pen = 1000.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                              np.sum(y_low ** 2) + np.sum(y_high ** 2))

        # Pairwise overlaps
        dx = xz[:, None] - xz[None, :]
        dy = yz[:, None] - yz[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        r_sum = rz[iu[0]] + rz[iu[1]]
        overlap = np.maximum(r_sum - d_pairs, 0.0)
        overlap_pen = 1000.0 * np.sum(overlap ** 2)

        # Small radius penalty
        small_pen = 500.0 * np.sum(np.maximum(0.005 - rz, 0.0) ** 2)

        return obj + wall_pen + overlap_pen + small_pen

    z0 = np.concatenate([x, y, r, [w]])
    bounds = []
    for i in range(N):
        bounds.append((r[i] * 0.5, w - r[i] * 0.5))
    for i in range(N):
        bounds.append((r[i] * 0.5, max(r[i] * 0.5, h - r[i] * 0.5)))
    for i in range(N):
        bounds.append((0.005, 0.4))
    bounds.append((0.4, 1.6))

    remaining = time_budget - (time.time() - start)
    if remaining < 2:
        return None

    maxiter = int(min(400, max(50, remaining * 20)))
    try:
        res = minimize(epigraph_obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        r_opt = res.x[2*N:3*N]
        w_opt = float(np.clip(res.x[3*N], 0.4, 1.6))

        if np.any(r_opt < 0.003):
            return None

        if _is_valid(x_opt, y_opt, r_opt, w_opt):
            return x_opt, y_opt, r_opt, w_opt

        # Try force relax
        x_f, y_f = _force_relax_unified(x_opt, y_opt, r_opt, w_opt, max_iter=300)
        if _is_valid(x_f, y_f, r_opt, w_opt):
            return x_f, y_f, r_opt, w_opt
    except Exception:
        pass
    return None


def _granular_polish(x, y, r, w, time_budget=10.0):
    """Fine-grained local search: try small perturbations and keep best."""
    h = 2.0 - w
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    rng = np.random.default_rng(12345)

    for _ in range(50):
        if time.time() - start > time_budget:
            break
        # Random small perturbation
        scale = 0.005
        x_p = x + rng.normal(0, scale, N)
        y_p = y + rng.normal(0, scale, N)
        x_p = np.clip(x_p, r, w - r)
        y_p = np.clip(y_p, r, h - r)

        x_t, y_t = _force_relax_unified(x_p, y_p, r, w, max_iter=150, tol=1e-10)
        if _is_valid(x_t, y_t, r, w):
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x_t.copy(), y_t.copy(), r.copy()

    return best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(x, y, r, w, tol=1e-8):
        return _is_valid(x, y, r, w, tol)

    def is_valid_equal_r(x, y, r, w, tol=1e-8):
        return _is_valid_equal_r(x, y, r, w, tol)

    def sum_radii(r):
        return float(np.sum(r))

    def equal_r_objective(z, r, w):
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w
        pen = 0.0
        x_low = np.maximum(r - x, 0.0)
        x_high = np.maximum(x - (w - r), 0.0)
        y_low = np.maximum(r - y, 0.0)
        y_high = np.maximum(y - (h - r), 0.0)
        pen += np.sum(x_low ** 2) + np.sum(x_high ** 2)
        pen += np.sum(y_low ** 2) + np.sum(y_high ** 2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        overlap = np.maximum(2 * r - d_pairs, 0.0)
        pen += np.sum(overlap ** 2)
        return pen

    def grid_init(w0, n=N):
        h0 = 2.0 - w0
        best_val = -1
        best_x = best_y = best_r = None

        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < n:
                    continue
                if rows * cols > n + 8:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_cell = min(cell_w, cell_h) / 2.0 * 0.98

                if r_cell < 0.015:
                    continue

                x = np.zeros(n)
                y = np.zeros(n)
                r = np.full(n, r_cell)

                idx = 0
                for ri in range(rows):
                    for ci in range(cols):
                        if idx >= n:
                            break
                        cx = (ci + 0.5) * cell_w
                        cy = (ri + 0.5) * cell_h
                        x[idx] = cx
                        y[idx] = cy
                        idx += 1

                if idx < n:
                    for i in range(idx, n):
                        x[i] = w0 / 2.0
                        y[i] = h0 / 2.0
                        r[i] = r_cell * 0.5

                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()

        if best_x is None:
            cols = 5
            rows = 5
            cell_w = w0 / cols
            cell_h = h0 / rows
            r0 = min(cell_w, cell_h) / 2.0 * 0.9
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.full(n, r0)
            for i in range(n):
                ri, ci = divmod(i, cols)
                x[i] = (ci + 0.5) * cell_w
                y[i] = (ri + 0.5) * cell_h
            best_x, best_y, best_r = x, y, r

        return best_x, best_y, best_r

    def find_positions_for_radius(r, w, x_init, y_init, max_iter=200):
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            res = minimize(equal_r_objective, z0, args=(r, w), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            return x_opt, y_opt, res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

    def hex_init(w0, n=N):
        h0 = 2.0 - w0
        area = w0 * h0
        r_est = np.sqrt(area / (n * np.pi / 0.9069))
        r_est = min(r_est, 0.15)

        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, r_est)

        dx = 2.0 * r_est * 1.002
        dy = np.sqrt(3) * r_est * 1.002
        row = 0
        col = 0
        idx = 0
        while idx < n:
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1
                col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[idx] = cx
            y[idx] = cy
            idx += 1
            col += 1

        for i in range(idx, n):
            x[i] = w0 / 2.0
            y[i] = h0 / 2.0
            r[i] = r_est * 0.6

        return x, y, r

    def binary_search_max_radius(w, x_init, y_init, r_low, r_high, max_iter=12, tol=1e-6):
        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_low
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_mid = (r_low + r_high) / 2.0
            if r_mid - r_low < tol:
                break
            x_opt, y_opt, obj_val = find_positions_for_radius(r_mid, w, x_init, y_init, max_iter=150)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_mid, w):
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_mid
                r_low = r_mid
            else:
                r_high = r_mid
        return best_x, best_y, best_r

    def force_relax(x, y, r, w, max_iter=800, tol=1e-10):
        return _force_relax_unified(x, y, r, w, max_iter=max_iter, tol=tol)

    def smooth_objective(z, penalty_weight=500.0):
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            return 1e8

        r_clipped = np.clip(r, 0.001, min(w, h) / 2.0 - 0.001)
        obj = -np.sum(r_clipped)

        # Wall violations
        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)
        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        # Pairwise overlaps (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        r_sum_pairs = r_clipped[iu[0]] + r_clipped[iu[1]]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def equal_r_slsqp(x_init, y_init, r_init, w_init, time_limit=15.0):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        r_target = r_init[0]
        h_init = 2.0 - w_init

        def obj(z):
            x = z[:N]
            y = z[N:2*N]
            w = z[2*N]
            h = 2.0 - w
            if w < 0.3 or h < 0.3:
                return 1e8
            x_low = np.maximum(r_target - x, 0.0)
            x_high = np.maximum(x - (w - r_target), 0.0)
            y_low = np.maximum(r_target - y, 0.0)
            y_high = np.maximum(y - (h - r_target), 0.0)
            wall_pen = 500.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                               np.sum(y_low ** 2) + np.sum(y_high ** 2))
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            iu = np.triu_indices(N, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
            overlap = np.maximum(2 * r_target - d_pairs, 0.0)
            overlap_pen = 500.0 * np.sum(overlap ** 2)
            space_reward = 0.1 * (w * (2.0 - w))
            return -space_reward + wall_pen + overlap_pen

        z0 = np.concatenate([x_init, y_init, [w_init]])
        bounds = [(r_target, w_init - r_target)] * N + \
                 [(r_target, h_init - r_target)] * N + [(0.3, 1.7)]
        try:
            maxiter = int(min(300, max(50, remaining * 20)))
            res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            w_opt = float(np.clip(res.x[2*N], 0.3, 1.7))
            if is_valid_equal_r(x_opt, y_opt, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_opt, y_opt, r_arr, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, np.full(N, r_target), w_opt, max_iter=100)
            if is_valid_equal_r(x_f, y_f, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_f, y_f, r_arr, w_opt
        except Exception:
            pass
        return None

    def smooth_slsqp(x_init, y_init, r_init, w_init, time_limit=18.0):
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        maxiter = int(min(500, max(100, remaining * 25)))
        h_init = 2.0 - w_init
        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.3, w_init - r_init[i] * 0.3))
        for i in range(N):
            bounds.append((r_init[i] * 0.3, max(r_init[i] * 0.3, h_init - r_init[i] * 0.3)))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))
        try:
            res = minimize(smooth_objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            if np.any(r_opt < 0.005):
                return None
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=200)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt
        except Exception:
            pass
        return None

    def grow_equal_radius(x, y, r, w, max_rounds=10):
        r_cur = r[0]
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            r_new = r_cur * 1.005
            x_opt, y_opt, obj_val = find_positions_for_radius(r_new, w, x, y, max_iter=100)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_new, w):
                x, y, r_cur = x_opt, y_opt, r_new
            else:
                break
        r_arr = np.full(N, r_cur)
        return x, y, r_arr, w

    def relax_positions(x, y, r, w, max_iter=300, ftol=1e-14):
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]
        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 50})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    def try_grow(x, y, r, w, max_rounds=5):
        h = 2.0 - w
        for round_idx in range(max_rounds):
            if (time.time() - start_time) > deadline - 8:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 8:
                    break
                for _ in range(50):
                    r_new = r.copy()
                    r_new[i] *= 1.003
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=40, ftol=1e-12)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
            if not improved:
                break
        return x, y, r, w

    def polish(x, y, r, w):
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            x_t, y_t = res.x[:N], res.x[N:2*N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w

    def newton_contact_solve(x_init, y_init, r, w, max_iter=50, tol=1e-12):
        """Newton-Raphson solver for exact contact configuration."""
        return _constraint_refine(x_init, y_init, r, w, max_iter=max_iter, tol=tol)

    def high_precision_polish(x, y, r, w, max_iter=200):
        """High-precision polish using Newton on the exact constraint system
        with very tight tolerances."""
        x_new, y_new = newton_contact_solve(x, y, r, w, max_iter=max_iter, tol=1e-14)
        if is_valid(x_new, y_new, r, w):
            return x_new, y_new
        return x, y

    def logsumexp_objective(z, r_fixed, w, weight=100.0):
        """Log-sum-exp smooth objective for equal radii positions."""
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w

        # Wall violations
        x_low = np.maximum(r_fixed - x, 0.0)
        x_high = np.maximum(x - (w - r_fixed), 0.0)
        y_low = np.maximum(r_fixed - y, 0.0)
        y_high = np.maximum(y - (h - r_fixed), 0.0)

        # Pairwise overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        overlap = np.maximum(2 * r_fixed - d_pairs, 0.0)

        # Log-sum-exp of squared violations
        all_viol = np.concatenate([x_low**2, x_high**2, y_low**2, y_high**2, overlap**2])
        max_v = np.max(all_viol)
        if max_v < 1e-14:
            return 0.0
        lse = max_v + np.log(np.sum(np.exp(all_viol - max_v)))

        return weight * lse

    def logsumexp_optimize(x_init, y_init, r, w, time_limit=12.0):
        """Optimize positions using log-sum-exp objective."""
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            maxiter = int(min(400, max(80, remaining * 30)))
            res = minimize(logsumexp_objective, z0, args=(r, w, 100.0), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 60})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            if is_valid_equal_r(x_opt, y_opt, r, w):
                return x_opt, y_opt, np.full(N, r), w
        except Exception:
            pass
        return None

    def multi_start_perturb(x, y, r, w, rng, n_perturb=3, scale=0.01):
        """Try multiple perturbations and keep the best valid result."""
        best_val = sum_radii(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        for _ in range(n_perturb):
            if time.time() > deadline - 10:
                break
            x_p = x + rng.normal(0, scale, N)
            y_p = y + rng.normal(0, scale, N)
            x_p = np.clip(x_p, r, w - r)
            h = 2.0 - w
            y_p = np.clip(y_p, r, h - r)

            x_t, y_t = force_relax(x_p, y_p, r, w, max_iter=300)
            if is_valid(x_t, y_t, r, w):
                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r = x_t.copy(), y_t.copy(), r.copy()

        return best_x, best_y, best_r

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: Try multiple aspect ratios with grid and hex initialization
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        # Grid init
        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=500)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

        # Hex init
        x0h, y0h, r0h = hex_init(w0)
        x_th, y_th = force_relax(x0h, y0h, r0h, w0, max_iter=500)
        if is_valid(x_th, y_th, r0h, w0):
            val_h = sum_radii(r0h)
            if val_h > best_value:
                best_value = val_h
                best_result = (x_th.copy(), y_th.copy(), r0h.copy(), w0)

        # Equal-radius binary search from this init
        if time.time() < deadline - 35:
            r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
            x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.9, r_init_val * 1.15, max_iter=10)
            if r_eq > 0.01:
                r_arr = np.full(N, r_eq)
                val_eq = sum_radii(r_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)

    # Phase 2: Equal-radius SLSQP from best
    if best_result is not None and time.time() < deadline - 30:
        x_cur, y_cur, r_cur, w_cur = best_result
        result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=18.0)
        if result_eq is not None:
            x_s, y_s, r_s, w_s = result_eq
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        # Log-sum-exp optimization
        if time.time() < deadline - 20:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            result_lse = logsumexp_optimize(x_cur, y_cur, r_eq, w_cur, time_limit=10.0)
            if result_lse is not None:
                x_s, y_s, r_s, w_s = result_lse
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        # Also try standard SLSQP
        if time.time() < deadline - 15:
            x_cur, y_cur, r_cur, w_cur = best_result
            result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=12.0)
            if result is not None:
                x_s, y_s, r_s, w_s = result
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 3: Grow radii from best
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result

        if np.std(r_cur) < 0.005:
            x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=8)
            if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                val_eq = sum_radii(r_eq_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)

        if time.time() < deadline - 12:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur)
            if is_valid(x_cur, y_cur, r_cur, w_cur):
                val = sum_radii(r_cur)
                if val > best_value:
                    best_value = val
                    best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        # Expansion relaxation polish (vectorized, targeted growth)
        if time.time() < deadline - 10:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_exp, y_exp, r_exp = _expansion_relaxation_polish(
                x_cur, y_cur, r_cur, w_cur, max_rounds=25, time_budget=12.0)
            if _is_valid(x_exp, y_exp, r_exp, w_cur):
                val_exp = sum_radii(r_exp)
                if val_exp > best_value:
                    best_value = val_exp
                    best_result = (x_exp.copy(), y_exp.copy(), r_exp.copy(), w_cur)

        # Smooth epigraph polish
        if time.time() < deadline - 8:
            x_cur, y_cur, r_cur, w_cur = best_result
            result_epi = _smooth_epigraph_polish(x_cur, y_cur, r_cur, w_cur, time_budget=10.0)
            if result_epi is not None:
                x_s, y_s, r_s, w_s = result_epi
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 4: Multi-start perturbation
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_p, y_p, r_p = multi_start_perturb(x_cur, y_cur, r_cur, w_cur, rng, n_perturb=4, scale=0.012)
        if is_valid(x_p, y_p, r_p, w_cur):
            val_p = sum_radii(r_p)
            if val_p > best_value:
                best_value = val_p
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_cur)

    # Phase 5: Polish
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        # Newton contact-graph solve for exact configuration
        if time.time() < deadline - 12:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_new, y_new = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=80, tol=1e-12)
            if is_valid(x_new, y_new, r_cur, w_cur):
                val_new = sum_radii(r_cur)
                if val_new > best_value:
                    best_value = val_new
                    best_result = (x_new.copy(), y_new.copy(), r_cur.copy(), w_cur)

    # Phase 6: Second optimization pass
    if best_result is not None and time.time() < deadline - 25:
        x_cur, y_cur, r_cur, w_cur = best_result

        result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=15.0)
        if result_eq is not None:
            x_s, y_s, r_s, w_s = result_eq
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        if time.time() < deadline - 10:
            x_cur, y_cur, r_cur, w_cur = best_result
            result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=10.0)
            if result is not None:
                x_s, y_s, r_s, w_s = result
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        # Log-sum-exp second pass
        if time.time() < deadline - 8:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            result_lse = logsumexp_optimize(x_cur, y_cur, r_eq, w_cur, time_limit=8.0)
            if result_lse is not None:
                x_s, y_s, r_s, w_s = result_lse
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 7: Additional random restarts
    if time.time() < deadline - 25:
        for _ in range(10):
            if time.time() > deadline - 20:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue

            if rng.random() < 0.5:
                x0, y0, r0 = grid_init(w0)
            else:
                x0, y0, r0 = hex_init(w0)

            x0 += rng.normal(0, 0.008, N)
            y0 += rng.normal(0, 0.008, N)

            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)

            if val > best_value * 0.92:
                if time.time() < deadline - 18:
                    r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
                    x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.1, max_iter=8)
                    if r_eq > 0.01:
                        r_arr = np.full(N, r_eq)
                        val_eq = sum_radii(r_arr)
                        if val_eq > best_value:
                            best_value = val_eq
                            best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)
                            continue

                if time.time() < deadline - 12:
                    result = smooth_slsqp(x_t, y_t, r0, w0, time_limit=8.0)
                    if result is not None:
                        x_s, y_s, r_s, w_s = result
                        val_s = sum_radii(r_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
                            continue

                if time.time() < deadline - 8:
                    x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0, max_rounds=3)
                    if is_valid(x_g, y_g, r_g, w_g):
                        val_g = sum_radii(r_g)
                        if val_g > best_value:
                            best_value = val_g
                            best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g)

    # Phase 8: Final polish and growth
    if best_result is not None and time.time() < deadline - 8:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        # Newton high-precision polish
        if time.time() < deadline - 5:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_hp, y_hp = high_precision_polish(x_cur, y_cur, r_cur, w_cur, max_iter=150)
            if is_valid(x_hp, y_hp, r_cur, w_cur):
                val_hp = sum_radii(r_cur)
                if val_hp > best_value:
                    best_value = val_hp
                    best_result = (x_hp.copy(), y_hp.copy(), r_cur.copy(), w_cur)

        # Granular polish: fine local search
        if time.time() < deadline - 4:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_gran, y_gran, r_gran = _granular_polish(x_cur, y_cur, r_cur, w_cur, time_budget=6.0)
            if _is_valid(x_gran, y_gran, r_gran, w_cur):
                val_gran = sum_radii(r_gran)
                if val_gran > best_value:
                    best_value = val_gran
                    best_result = (x_gran.copy(), y_gran.copy(), r_gran.copy(), w_cur)

        if time.time() < deadline - 3:
            x_cur, y_cur, r_cur, w_cur = best_result
            if np.std(r_cur) < 0.005:
                x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=5)
                if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                    val_eq = sum_radii(r_eq_arr)
                    if val_eq > best_value:
                        best_value = val_eq
                        best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)
            else:
                x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur, max_rounds=3)
                if is_valid(x_cur, y_cur, r_cur, w_cur):
                    val = sum_radii(r_cur)
                    if val > best_value:
                        best_value = val
                        best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

            # Final Newton solve after growth
            if time.time() < deadline - 2:
                x_cur, y_cur, r_cur, w_cur = best_result
                x_fn, y_fn = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=60, tol=1e-13)
                if is_valid(x_fn, y_fn, r_cur, w_cur):
                    val_fn = sum_radii(r_cur)
                    if val_fn > best_value:
                        best_value = val_fn
                        best_result = (x_fn.copy(), y_fn.copy(), r_cur.copy(), w_cur)

    if best_result is None:
        w0 = 1.0
        h0 = 1.0
        cols = 5
        rows = 5
        cell_w = w0 / cols
        cell_h = h0 / rows
        r0 = min(cell_w, cell_h) / 2.0 * 0.45
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            cells.append((cx, cy, r0))
        return np.array(cells)

    x_final, y_final, r_final, w_final = best_result

    # Safety shrink
    for _ in range(10):
        r_trial = r_final * (1 - 1e-10)
        if is_valid(x_final, y_final, r_trial, w_final):
            r_final = r_trial
        else:
            break

    # Final Newton solve on the best result
    if time.time() < deadline - 1:
        x_final, y_final = newton_contact_solve(x_final, y_final, r_final, w_final, max_iter=40, tol=1e-13)
        if is_valid(x_final, y_final, r_final, w_final):
            pass
        else:
            # Revert if Newton made it invalid
            x_final, y_final, r_final, w_final = best_result

    # Final expansion relaxation if time permits
    if time.time() < deadline - 0.5:
        x_final, y_final, r_final = _expansion_relaxation_polish(
            x_final, y_final, r_final, w_final, max_rounds=10, time_budget=1.5)
        if _is_valid(x_final, y_final, r_final, w_final):
            pass
        else:
            x_final, y_final, r_final, w_final = best_result

    return np.column_stack((x_final, y_final, r_final))
