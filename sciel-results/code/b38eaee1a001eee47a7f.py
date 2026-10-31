"""Circle packing in perimeter-4 rectangle: Constraint-based SLSQP with force polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_value = 0.0
    best_result = None

    # --- Helper: check validity ---
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

    def _check_validity_z(z, tol=1e-8):
        """z layout: [w, r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]"""
        w = z[0]
        h = 2.0 - w
        r = z[1:1 + N]
        cx = z[1 + N:1 + 2 * N]
        cy = z[1 + 2 * N:1 + 3 * N]
        if not _is_valid(cx, cy, r, w, h, tol):
            return False, 0.0
        return True, float(np.sum(r))

    # --- Phase 1: Multi-start with penalty-based SLSQP ---
    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    penalty_weight = 500.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                dist_req = r[i] + r[j]
                if d < dist_req:
                    obj += penalty_weight * (dist_req - d) ** 2

        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2

        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def _force_relax(z, max_iter=300):
        z = z.copy().astype(float)
        n = N
        dt = 0.005
        min_dt = 0.0005

        for it in range(max_iter):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w

            fx = np.zeros(n)
            fy = np.zeros(n)

            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 25
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 25
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 25
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 25

            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 12
                        fy[i] += (dy / d) * overlap * 12
                        fx[j] -= (dx / d) * overlap * 12
                        fy[j] -= (dy / d) * overlap * 12

            max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
            if max_force > 0.5:
                dt = max(min_dt, 0.002)
            elif max_force > 0.1:
                dt = max(min_dt, 0.004)
            else:
                dt = 0.005

            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def get_initial_guesses():
        guesses = []
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
        side = int(np.ceil(np.sqrt(N)))

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        rng = np.random.default_rng(42)
        for _ in range(4):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        z0_relaxed = _force_relax(z0, max_iter=200)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(300, max(50, remaining * 30)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 60})

            z_opt = res.x
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w

            valid, val = _is_valid(x, y, r, w, h), float(np.sum(r))
            if valid and val > best_value:
                best_value = val
                # Convert to constraint layout: [w, r..., cx..., cy...]
                best_result = np.concatenate([[w], r, x, y])
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                x_r = z_repair[0::3][:N]
                y_r = z_repair[1::3][:N]
                r_r = z_repair[2::3][:N]
                w_r = z_repair[-1]
                h_r = 2.0 - w_r
                valid_r, val_r = _is_valid(x_r, y_r, r_r, w_r, h_r), float(np.sum(r_r))
                if valid_r and val_r > best_value:
                    best_value = val_r
                    best_result = np.concatenate([[w_r], r_r, x_r, y_r])

        except Exception:
            continue

    # --- Phase 2: Constraint-based SLSQP refinement on best result ---
    if best_result is not None and time.time() < deadline - 25:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time))
        if refined is not None:
            valid, val = _check_validity_z(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # --- Phase 3: Force-directed polish + radius growth ---
    if best_result is not None and time.time() < deadline - 10:
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = _check_validity_z(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # --- Phase 4: Final high-precision constraint refinement ---
    if best_result is not None and time.time() < deadline - 5:
        refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
        if refined2 is not None:
            valid, val = _check_validity_z(refined2)
            if valid and val > best_value:
                best_value = val
                best_result = refined2

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output: [w, r_0..r_N, cx_0..cx_N, cy_0..cy_N] -> (x, y, r)
    w = best_result[0]
    r = best_result[1:1 + N]
    x = best_result[1 + N:1 + 2 * N]
    y = best_result[1 + 2 * N:1 + 3 * N]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(50):
        r_trial = r * (1 - 1e-7)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _constraint_refine(z, time_budget):
    """SLSQP with explicit constraints. z layout: [w, r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]"""
    z = z.copy().astype(float)
    n = N

    def obj(zv):
        return -np.sum(zv[1:1 + n])

    constraints = []
    bounds = [(0.3, 1.7)]  # w
    for i in range(n):
        bounds.append((0.001, 1.0))  # r
    for i in range(n):
        bounds.append((0.0, 2.0))  # cx
    for i in range(n):
        bounds.append((0.0, 2.0))  # cy

    # Boundary constraints
    for i in range(n):
        r_idx = 1 + i
        cx_idx = 1 + n + i
        cy_idx = 1 + 2 * n + i
        constraints.append({'type': 'ineq', 'fun': lambda p, cx=cx_idx, r=r_idx: p[cx] - p[r]})
        constraints.append({'type': 'ineq', 'fun': lambda p, w=0, cx=cx_idx, r=r_idx: p[w] - p[cx] - p[r]})
        constraints.append({'type': 'ineq', 'fun': lambda p, cy=cy_idx, r=r_idx: p[cy] - p[r]})
        constraints.append({'type': 'ineq', 'fun': lambda p, w=0, cy=cy_idx, r=r_idx: (2.0 - p[w]) - p[cy] - p[r]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            ri = 1 + i
            rj = 1 + j
            cxi = 1 + n + i
            cxj = 1 + n + j
            cyi = 1 + 2 * n + i
            cyj = 1 + 2 * n + j

            def make_overlap(ri_, rj_, cxi_, cxj_, cyi_, cyj_):
                def fun(p):
                    dx = p[cxi_] - p[cxj_]
                    dy = p[cyi_] - p[cyj_]
                    return dx * dx + dy * dy - (p[ri_] + p[rj_]) ** 2
                return fun

            constraints.append({'type': 'ineq', 'fun': make_overlap(ri, rj, cxi, cxj, cyi, cyj)})

    maxiter = int(min(400, max(50, time_budget * 80)))
    try:
        res = minimize(obj, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

        z_opt = res.x
        w = z_opt[0]
        h = 2.0 - w
        r = z_opt[1:1 + n]
        cx = z_opt[1 + n:1 + 2 * n]
        cy = z_opt[1 + 2 * n:1 + 3 * n]

        if _is_valid(cx, cy, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _polish_result(z, time_budget):
    """Force-directed position polish + radius growth. z layout: [w, r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]"""
    z = z.copy().astype(float)
    n = N
    w = z[0]
    h = 2.0 - w
    r = z[1:1 + n].copy()
    x = z[1 + n:1 + 2 * n].copy()
    y = z[1 + 2 * n:1 + 3 * n].copy()

    dt = 0.001
    max_iter = int(min(1000, time_budget * 60))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 35
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 35
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 35
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 35

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 18
                    fy[i] += (dy / d) * overlap * 18
                    fx[j] -= (dx / d) * overlap * 18
                    fy[j] -= (dy / d) * overlap * 18

        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii
    grow_factor = 1.0003
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0] = w
    z_new[1:1 + n] = r
    z_new[1 + n:1 + 2 * n] = x
    z_new[1 + 2 * n:1 + 3 * n] = y

    if _is_valid(x, y, r, w, h):
        return z_new
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
