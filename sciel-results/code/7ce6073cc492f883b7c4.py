"""Circle packing in perimeter-4 rectangle: SLSQP + Epigraph + Constraint + Greedy Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_LSE_K = 500.0


def _vectorized_objective(z, penalty_weight=1000.0):
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

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _lse(vals):
    m = np.max(vals)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(vals - m)))


def construct():
    start_time = time.time()
    deadline = start_time + 143.0

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
        return True, float(np.sum(r))

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
                z[3 * i] = (col + 0.5) / side * w0
                z[3 * i + 1] = (row + 0.5) / side * h0
                z[3 * i + 2] = min(w0 / side, h0 / side) * 0.45
            z[-1] = w0
            guesses.append(z)
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                z[3 * i] = rng.uniform(r0, w0 - r0)
                z[3 * i + 1] = rng.uniform(r0, h0 - r0)
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 22:
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
            continue

    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 22:
            expanded = _expansion_relaxation_polish_enhanced(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        if best_result is not None and time.time() < deadline - 18:
            ep = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 10)
            if ep is not None:
                valid, val = check_validity(ep)
                if valid and val > best_value:
                    best_value = val
                    best_result = ep

        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 3:
            greedy = _greedy_expansion(best_result, deadline - (time.time() - start_time))
            if greedy is not None:
                valid, val = check_validity(greedy)
                if valid and val > best_value:
                    best_value = val
                    best_result = greedy

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

    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    conv_tol = 1e-8
    iu = np.triu_indices(n, k=1)
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
        fx -= np.maximum(r - x, 0.0) * 100
        fx += np.maximum(x + r - w, 0.0) * 100
        fy -= np.maximum(r - y, 0.0) * 100
        fy += np.maximum(y + r - h, 0.0) * 100
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size else 0.0
        if max_overlap > conv_tol:
            f_pairs = overlap * 15.0
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            for idx in range(len(iu[0])):
                ii, jj = iu[0][idx], iu[1][idx]
                fx[ii] += ux[idx] * f_pairs[idx]
                fy[ii] += uy[idx] * f_pairs[idx]
                fx[jj] -= ux[idx] * f_pairs[idx]
                fy[jj] -= uy[idx] * f_pairs[idx]
        if max_overlap < conv_tol:
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
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    iu = np.triu_indices(n, k=1)
    max_iter = int(min(3000, time_budget * 150))
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        fx -= np.maximum(r - x, 0.0) * 35
        fx += np.maximum(x + r - w, 0.0) * 35
        fy -= np.maximum(r - y, 0.0) * 35
        fy += np.maximum(y + r - h, 0.0) * 35
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            for idx in range(len(iu[0])):
                ii, jj = iu[0][idx], iu[1][idx]
                fx[ii] += ux[idx] * f_pairs[idx]
                fy[ii] += uy[idx] * f_pairs[idx]
                fx[jj] -= ux[idx] * f_pairs[idx]
                fy[jj] -= uy[idx] * f_pairs[idx]
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        dt = 0.0005 if max_force > 0.1 else 0.001
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    grow = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        rt = r * grow
        if _is_valid(x, y, rt, w, h):
            r = rt
        else:
            break
    if _is_valid(x, y, r, w, h):
        zn = np.empty(3 * N + 1)
        zn[0::3][:n] = x
        zn[1::3][:n] = y
        zn[2::3][:n] = r
        zn[-1] = w
        return zn
    return None


def _expansion_relaxation_polish_enhanced(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071),
        (0.866, 0.5), (-0.866, 0.5), (0.866, -0.5), (-0.866, -0.5)
    ]
    for rnd in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                ra = d - r[j]
                if ra < r_max:
                    r_max = ra
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-6:
                lo, hi = r[i], r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    rtmp = r.copy()
                    rtmp[i] = mid
                    if _is_valid(x, y, rtmp, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
            min_gap = float('inf')
            bd, by = 0.0, 0.0
            if x[i] - r[i] < min_gap:
                min_gap, bd, by = x[i] - r[i], 1.0, 0.0
            if w - x[i] - r[i] < min_gap:
                min_gap, bd, by = w - x[i] - r[i], -1.0, 0.0
            if y[i] - r[i] < min_gap:
                min_gap, bd, by = y[i] - r[i], 0.0, 1.0
            if h - y[i] - r[i] < min_gap:
                min_gap, bd, by = h - y[i] - r[i], 0.0, -1.0
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        bd, by = (x[i] - x[j]) / d, (y[i] - y[j]) / d
            if min_gap < 0.05:
                found = False
                for dx_dir, dy_dir in directions:
                    if found:
                        break
                    for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
                        nx, ny = x[i] + dx_dir * step, y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, nx, ny):
                            old_rm = _max_radius_for_circle(x, y, r, w, h, i)
                            x[i], y[i] = nx, ny
                            new_rm = _max_radius_for_circle(x, y, r, w, h, i)
                            if new_rm > old_rm + 1e-6:
                                improved = True
                                found = True
                                break
                            else:
                                x[i], y[i] = nx - dx_dir * step, ny - dy_dir * step
        if not improved:
            break
    if time.time() < time_budget - 5:
        for rnd in range(30):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(n):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    ra = d - r[j]
                    if ra < r_max:
                        r_max = ra
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-5:
                    lo, hi = r[i], r_max
                    for _ in range(35):
                        mid = (lo + hi) / 2.0
                        rtmp = r.copy()
                        rtmp[i] = mid
                        if _is_valid(x, y, rtmp, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-6:
                        r[i] = lo
                        improved = True
            if not improved:
                break
    grow = 1.0005
    for _ in range(300):
        if time.time() > time_budget:
            break
        rt = r * grow
        if _is_valid(x, y, rt, w, h):
            r = rt
        else:
            break
    if _is_valid(x, y, r, w, h):
        zn = np.empty(3 * N + 1)
        zn[0::3][:n] = x
        zn[1::3][:n] = y
        zn[2::3][:n] = r
        zn[-1] = w
        return zn
    return None


def _max_radius_for_circle(x, y, r, w, h, i):
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        ra = d - r[j]
        if ra < r_max:
            r_max = ra
    return max(0.0, r_max)


def _is_valid_move(x, y, r, w, h, i, nx, ny):
    if nx < r[i] or nx > w - r[i]:
        return False
    if ny < r[i] or ny > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx, dy = nx - x[j], ny - y[j]
        rs = r[i] + r[j]
        if dx * dx + dy * dy < rs * rs - 1e-9:
            return False
    return True


def _epigraph_refine(z, time_budget):
    z = z.copy().astype(float)
    n = N
    K = _LSE_K
    iu = np.triu_indices(n, k=1)
    pairs = iu

    def obj(p):
        x = p[0::3][:n]
        y = p[1::3][:n]
        r = p[2::3][:n]
        t = p[-2]
        w = p[-1]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        dx = x[pairs[0]] - x[pairs[1]]
        dy = y[pairs[0]] - y[pairs[1]]
        d = np.sqrt(dx * dx + dy * dy)
        r_sum = r[pairs[0]] + r[pairs[1]]
        pair_viol = np.maximum(r_sum - d, 0.0)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        bnd = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
        allv = np.concatenate([pair_viol, bnd])
        lse = _lse(allv)
        return t - np.sum(r) + lse / K

    x0 = z[0::3][:n]
    y0 = z[1::3][:n]
    r0 = z[2::3][:n]
    w0 = z[-1]
    h0 = 2.0 - w0
    dmin = 1e9
    for a in range(n):
        for b in range(a + 1, n):
            dmin = min(dmin, np.hypot(x0[a] - x0[b], y0[a] - y0[b]))
    t0 = max(0.0, dmin - 1e-3)

    p0 = np.empty(3 * n + 2)
    p0[0::3][:n] = x0
    p0[1::3][:n] = y0
    p0[2::3][:n] = r0
    p0[-2] = t0
    p0[-1] = w0

    bounds = [(0, 2)] * n + [(0, 2)] * n + [(0.001, 1.0)] * n + [(0.0, 2.0)] + [(0.3, 1.7)]
    maxiter = int(min(1500, time_budget * 250))
    try:
        res = minimize(obj, p0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 120})
        popt = res.x
        x = popt[0::3][:n]
        y = popt[1::3][:n]
        r = popt[2::3][:n]
        w = popt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            zn = np.empty(3 * N + 1)
            zn[0::3][:n] = x
            zn[1::3][:n] = y
            zn[2::3][:n] = r
            zn[-1] = w
            return zn
        return None
    except Exception:
        return None


def _constraint_refine(z, time_budget):
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

    def bcon(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': bcon(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': bcon(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': bcon(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': bcon(i, 1, 1)})
    for i in range(n):
        for j in range(i + 1, n):
            def oc(i1, i2):
                return lambda p: (
                    (p[3 * i1] - p[3 * i2]) ** 2 +
                    (p[3 * i1 + 1] - p[3 * i2 + 1]) ** 2 -
                    (p[3 * i1 + 2] + p[3 * i2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': oc(i, j)})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        zopt = res.x
        x = zopt[0::3][:n]
        y = zopt[1::3][:n]
        r = zopt[2::3][:n]
        w = zopt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return zopt
        return None
    except Exception:
        return None


def _coordinate_ascent_refine(x, y, r, w, h, time_budget, max_rounds=20):
    """Per-circle coordinate ascent: move each circle to maximize its own radius."""
    n = N
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Try moving circle i in all 8 directions
            for angle in range(8):
                theta = angle * np.pi / 4.0
                for step in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                    dx = step * np.cos(theta)
                    dy = step * np.sin(theta)
                    nx, ny = x[i] + dx, y[i] + dy
                    if _is_valid_move(x, y, r, w, h, i, nx, ny):
                        old_rm = _max_radius_for_circle(x, y, r, w, h, i)
                        x[i], y[i] = nx, ny
                        new_rm = _max_radius_for_circle(x, y, r, w, h, i)
                        if new_rm > old_rm + 1e-7:
                            improved = True
                            # Expand radius to the new max
                            lo, hi = r[i], new_rm
                            for _ in range(25):
                                mid = (lo + hi) / 2.0
                                rtmp = r.copy()
                                rtmp[i] = mid
                                if _is_valid(x, y, rtmp, w, h):
                                    lo = mid
                                else:
                                    hi = mid
                            if lo > r[i] + 1e-8:
                                r[i] = lo
                            break
                        else:
                            x[i], y[i] = nx - dx, ny - dy
                    else:
                        break
        if not improved:
            break
    return x, y, r


def _adaptive_radius_polish(x, y, r, w, h, time_budget, max_rounds=15):
    """Adaptive per-circle radius growth with position adjustment."""
    n = N
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Try growing radius i by small amount
            for grow_factor in [1.002, 1.001, 1.0005, 1.0002, 1.0001]:
                new_r = r[i] * grow_factor
                if new_r <= r[i] + 1e-9:
                    continue
                # Check if we can fit with current position
                rt = r.copy()
                rt[i] = new_r
                if _is_valid(x, y, rt, w, h):
                    r[i] = new_r
                    improved = True
                    break
                else:
                    # Try to move circle to make room
                    # Find the direction that maximizes clearance
                    best_pos = None
                    best_clearance = -1e9
                    for angle in range(16):
                        theta = angle * 2 * np.pi / 16
                        for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                            nx = x[i] + step * np.cos(theta)
                            ny = y[i] + step * np.sin(theta)
                            if _is_valid_move(x, y, rt, w, h, i, nx, ny):
                                rt2 = r.copy()
                                rt2[i] = new_r
                                if _is_valid(np.where(np.arange(N) == i, nx, x),
                                             np.where(np.arange(N) == i, ny, y),
                                             rt2, w, h):
                                    clearance = _max_radius_for_circle(
                                        np.where(np.arange(N) == i, nx, x),
                                        np.where(np.arange(N) == i, ny, y),
                                        rt2, w, h, i)
                                    if clearance > best_clearance:
                                        best_clearance = clearance
                                        best_pos = (nx, ny)
                    if best_pos is not None and best_clearance > new_r - 1e-6:
                        x[i], y[i] = best_pos
                        r[i] = new_r
                        improved = True
                        break
            if time.time() > time_budget:
                break
        if not improved:
            break
    return x, y, r


def _greedy_expansion(z, time_budget):
    """Final per-circle greedy expansion with binary search for the true max radius."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    # Phase 1: Coordinate ascent to find better positions
    if time.time() < time_budget - 5:
        x, y, r = _coordinate_ascent_refine(x, y, r, w, h, time_budget - 5, max_rounds=10)
    
    # Phase 2: Adaptive radius polish
    if time.time() < time_budget - 3:
        x, y, r = _adaptive_radius_polish(x, y, r, w, h, time_budget - 3, max_rounds=10)
    
    # Phase 3: Standard greedy expansion
    for _ in range(3):
        if time.time() > time_budget:
            break
        for i in range(n):
            if time.time() > time_budget:
                break
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                ra = d - r[j]
                if ra < r_max:
                    r_max = ra
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                lo, hi = r[i], r_max
                for _ in range(40):
                    mid = (lo + hi) / 2.0
                    rt = r.copy()
                    rt[i] = mid
                    if _is_valid(x, y, rt, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
    if _is_valid(x, y, r, w, h):
        zn = np.empty(3 * N + 1)
        zn[0::3][:n] = x
        zn[1::3][:n] = y
        zn[2::3][:n] = r
        zn[-1] = w
        return zn
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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
