"""Circle packing in perimeter-4 rectangle: L-BFGS-B penalty + expansion polish + SLSQP refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    iu = np.triu_indices(N, k=1)

    def is_valid(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False
        if np.any(r <= 0):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False
        return True

    def force_relax(x, y, r, w, h, max_iter=2000, tol=1e-10):
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 50
            fx += right_viol * 50
            fy -= bottom_viol * 50
            fy += top_viol * 50
            dx = x[iu[0]] - x[iu[1]]
            dy = y[iu[0]] - y[iu[1]]
            d2 = dx * dx + dy * dy
            r_sum = r[iu[0]] + r[iu[1]]
            d = np.sqrt(np.maximum(d2, 1e-12))
            overlap = np.maximum(r_sum - d, 0.0)
            if np.max(overlap) < tol:
                break
            f_pairs = overlap * 10.0
            ux = dx / d
            uy = dy / d
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                fx[i] += ux[k] * f_pairs[k]
                fy[i] += uy[k] * f_pairs[k]
                fx[j] -= ux[k] * f_pairs[k]
                fy[j] -= uy[k] * f_pairs[k]
            max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
            dt = 0.005 if max_force > 0.1 else 0.002
            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
        return x, y, r

    def max_radius_for_circle(i, x, y, r, w, h):
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def is_valid_move(i, nx, ny, r_i, x, y, r, w, h, tol=1e-9):
        if nx < r_i - tol or nx > w - r_i + tol:
            return False
        if ny < r_i - tol or ny > h - r_i + tol:
            return False
        for j in range(N):
            if i == j:
                continue
            dx = nx - x[j]
            dy = ny - y[j]
            d2 = dx * dx + dy * dy
            rs = r_i + r[j]
            if d2 < rs * rs - tol:
                return False
        return True

    def expansion_polish(x, y, r, w, h, time_budget=5.0):
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        steps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]
        for outer in range(50):
            if time.time() - start > time_budget:
                break
            improved = False
            order = np.argsort(r)
            for i in order:
                if time.time() - start > time_budget:
                    break
                r_max = max_radius_for_circle(i, x, y, r, w, h)
                if r_max > r[i] + 1e-8:
                    lo = r[i]
                    hi = r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        r_test = r.copy()
                        r_test[i] = mid
                        if is_valid(x, y, r_test, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-9:
                        r[i] = lo
                        improved = True
                        continue
                best_move = None
                best_gain = 1e-6
                for angle in angles:
                    for step in steps:
                        nx = x[i] + step * np.cos(angle)
                        ny = y[i] + step * np.sin(angle)
                        if not is_valid_move(i, nx, ny, r[i], x, y, r, w, h):
                            continue
                        r_wall_new = min(nx, w - nx, ny, h - ny)
                        r_max_new = r_wall_new
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            d = np.sqrt(dx * dx + dy * dy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max_new:
                                r_max_new = r_allowed
                        r_max_new = max(0.0, r_max_new)
                        gain = r_max_new - r[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (nx, ny, r_max_new)
                if best_move is not None:
                    x[i] = best_move[0]
                    y[i] = best_move[1]
                    r[i] = best_move[2]
                    improved = True
            if not improved:
                break
        return x, y, r

    def lbfgs_optimize(z0, time_budget=10.0):
        x0 = z0[0::3][:N]
        y0 = z0[1::3][:N]
        r0 = z0[2::3][:N]
        w0 = z0[-1]
        h0 = 2.0 - w0

        def objective(zv):
            x = zv[0::3][:N]
            y = zv[1::3][:N]
            r = zv[2::3][:N]
            w = zv[-1]
            h = 2.0 - w
            if w < 0.1 or h < 0.1:
                return 1e6
            obj = -np.sum(r)
            dx = x[iu[0]] - x[iu[1]]
            dy = y[iu[0]] - y[iu[1]]
            d2 = dx * dx + dy * dy
            r_sum = r[iu[0]] + r[iu[1]]
            d = np.sqrt(np.maximum(d2, 0.0))
            overlap = np.maximum(r_sum - d, 0.0)
            obj += 1000.0 * np.sum(overlap * overlap)
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            obj += 1000.0 * (np.sum(left_viol * left_viol) +
                             np.sum(right_viol * right_viol) +
                             np.sum(bottom_viol * bottom_viol) +
                             np.sum(top_viol * top_viol))
            if w < 0.5:
                obj += 1000.0 * (0.5 - w) ** 2
            elif w > 1.5:
                obj += 1000.0 * (w - 1.5) ** 2
            return obj

        def gradient(zv):
            x = zv[0::3][:N]
            y = zv[1::3][:N]
            r = zv[2::3][:N]
            w = zv[-1]
            h = 2.0 - w
            if w < 0.1 or h < 0.1:
                return np.zeros_like(zv)
            grad = np.zeros_like(zv)
            r_idx = np.arange(2, 3 * N, 3)
            grad[r_idx] = -1.0
            dx = x[iu[0]] - x[iu[1]]
            dy = y[iu[0]] - y[iu[1]]
            d2 = dx * dx + dy * dy
            r_sum = r[iu[0]] + r[iu[1]]
            d = np.sqrt(np.maximum(d2, 1e-12))
            overlap = np.maximum(r_sum - d, 0.0)
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                if overlap[k] > 0:
                    factor = 2.0 * 1000.0 * overlap[k]
                    grad[3 * i] += factor * (x[i] - x[j]) / d[k]
                    grad[3 * j] -= factor * (x[i] - x[j]) / d[k]
                    grad[3 * i + 1] += factor * (y[i] - y[j]) / d[k]
                    grad[3 * j + 1] -= factor * (y[i] - y[j]) / d[k]
                    grad[3 * i + 2] += factor
                    grad[3 * j + 2] += factor
            for i in range(N):
                if r[i] > x[i]:
                    factor = 2.0 * 1000.0 * (r[i] - x[i])
                    grad[3 * i] -= factor
                    grad[3 * i + 2] += factor
                if x[i] + r[i] > w:
                    factor = 2.0 * 1000.0 * (x[i] + r[i] - w)
                    grad[3 * i] += factor
                    grad[3 * i + 2] += factor
                    grad[-1] -= factor
                if r[i] > y[i]:
                    factor = 2.0 * 1000.0 * (r[i] - y[i])
                    grad[3 * i + 1] -= factor
                    grad[3 * i + 2] += factor
                if y[i] + r[i] > h:
                    factor = 2.0 * 1000.0 * (y[i] + r[i] - h)
                    grad[3 * i + 1] += factor
                    grad[3 * i + 2] += factor
                    grad[-1] += factor
            return grad

        bounds = []
        for i in range(N):
            bounds.extend([(r0[i] * 0.3, min(x0[i], w0 - x0[i], y0[i], h0 - y0[i]) * 1.5 + 0.01),
                           (r0[i] * 0.3, min(y0[i], h0 - y0[i]) * 1.5 + 0.01),
                           (0.001, 0.3)])
        bounds.append((max(0.3, w0 - 0.3), min(1.7, w0 + 0.3)))

        maxiter = int(min(300, time_budget * 50))
        try:
            res = minimize(objective, z0, method='L-BFGS-B', jac=gradient,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50})
            z_opt = res.x
            xo = z_opt[0::3][:N]
            yo = z_opt[1::3][:N]
            ro = z_opt[2::3][:N]
            wo = z_opt[-1]
            ho = 2.0 - wo
            if is_valid(xo, yo, ro, wo, ho):
                return z_opt
        except Exception:
            pass
        return None

    def slsqp_refine(z, time_budget=10.0):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        def obj_slsqp(zv):
            rv = zv[2::3][:N]
            return -np.sum(rv)

        constraints = []
        bounds = []
        for i in range(N):
            bounds.extend([(r[i] * 0.5, min(x[i], w - x[i], y[i], h - y[i]) * 1.5),
                           (r[i] * 0.5, min(y[i], h - y[i]) * 1.5),
                           (0.001, 1.0)])
        bounds.append((max(0.3, w - 0.1), min(1.7, w + 0.1)))
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (p[3 * i] - p[3 * j]) ** 2 + (p[3 * i + 1] - p[3 * j + 1]) ** 2 - (p[3 * i + 2] + p[3 * j + 2]) ** 2
                })
        maxiter = int(min(300, time_budget * 60))
        try:
            res = minimize(obj_slsqp, z, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            xo = z_opt[0::3][:N]
            yo = z_opt[1::3][:N]
            ro = z_opt[2::3][:N]
            wo = z_opt[-1]
            ho = 2.0 - wo
            if is_valid(xo, yo, ro, wo, ho):
                return z_opt
        except Exception:
            pass
        return None

    def get_fallback():
        w = 1.0
        h = 1.0
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.09
        if not is_valid(x, y, r, w, h, tol=1e-6):
            r = r * 0.9
        return np.column_stack((x, y, r))

    def get_initial_guesses():
        guesses = []
        side = 5
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = min(w0 / side, h0 / side) * 0.4
            guesses.append((x, y, r, w0))
        for seed_val in range(10):
            rng = np.random.default_rng(seed_val * 1000 + 42)
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0))
        return guesses

    guesses = get_initial_guesses()

    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0

        x, y, r = force_relax(x0, y0, r0, w0, h0, max_iter=1000)

        z = np.empty(3 * N + 1)
        z[0::3][:N] = x
        z[1::3][:N] = y
        z[2::3][:N] = r
        z[-1] = w0

        # L-BFGS-B optimization
        remaining = deadline - time.time() - 25
        if remaining > 5:
            z_lbfgs = lbfgs_optimize(z, time_budget=min(8.0, remaining))
            if z_lbfgs is not None:
                z = z_lbfgs
                x = z[0::3][:N]
                y = z[1::3][:N]
                r = z[2::3][:N]
                w0 = z[-1]
                h0 = 2.0 - w0

        # Expansion polish
        remaining = deadline - time.time() - 20
        if remaining > 3:
            x, y, r = expansion_polish(x, y, r, w0, h0, time_budget=min(5.0, remaining))

        if is_valid(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w0
                best_result = z.copy()

        # SLSQP refinement
        remaining = deadline - time.time() - 15
        if remaining > 5 and best_result is not None:
            z_slsqp = slsqp_refine(z, time_budget=min(6.0, remaining))
            if z_slsqp is not None:
                z = z_slsqp
                x = z[0::3][:N]
                y = z[1::3][:N]
                r = z[2::3][:N]
                w0 = z[-1]
                h0 = 2.0 - w0
                if is_valid(x, y, r, w0, h0):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        best_result = z.copy()

                # Polish after SLSQP
                remaining = deadline - time.time() - 10
                if remaining > 3:
                    x, y, r = expansion_polish(x, y, r, w0, h0, time_budget=min(4.0, remaining))
                    if is_valid(x, y, r, w0, h0):
                        val = float(np.sum(r))
                        if val > best_value:
                            best_value = val
                            z = np.empty(3 * N + 1)
                            z[0::3][:N] = x
                            z[1::3][:N] = y
                            z[2::3][:N] = r
                            z[-1] = w0
                            best_result = z.copy()

    # Phase 2: Polish best result
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        x, y, r = expansion_polish(x, y, r, w, h, time_budget=min(10.0, deadline - time.time() - 8))
        if is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

    # Phase 3: Try different aspect ratios
    if best_result is not None and time.time() < deadline - 8:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w_orig = best_result[-1]
        h_orig = 2.0 - w_orig
        for w_try in [w_orig * 0.9, w_orig * 1.1, w_orig * 0.85, w_orig * 1.15]:
            if time.time() > deadline - 6:
                break
            if w_try < 0.3 or w_try > 1.7:
                continue
            h_try = 2.0 - w_try
            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy)
            xs, ys, rs = force_relax(xs, ys, rs, w_try, h_try, max_iter=500)
            xs, ys, rs = expansion_polish(xs, ys, rs, w_try, h_try, time_budget=min(3.0, deadline - time.time() - 4))
            if is_valid(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = xs
                    z[1::3][:N] = ys
                    z[2::3][:N] = rs
                    z[-1] = w_try
                    best_result = z.copy()

    if best_result is None:
        return get_fallback()

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w_final = best_result[-1]
    h_final = 2.0 - w_final

    lo, hi = 0.8, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if is_valid(x, y, r_trial, w_final, h_final, tol=1e-6):
            lo = mid
        else:
            hi = mid
    r_final = r * lo
    if not is_valid(x, y, r_final, w_final, h_final, tol=1e-6):
        return get_fallback()

    return np.column_stack((x, y, r_final))
