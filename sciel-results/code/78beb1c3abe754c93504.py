"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a different structural approach:
  1. Start from a compact hexagonal-style lattice.
  2. Use gradient-based optimization (L-BFGS-B) with smooth constraint penalties.
  3. Iteratively grow radii by solving a constrained sub-problem.
  4. Use simulated-annealing-like restarts on the aspect ratio.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def compute_value(x, y, r, w):
        """Return sum of radii if valid, else -inf."""
        h = 2.0 - w
        if w <= 1e-6 or h <= 1e-6:
            return -np.inf
        for i in range(N):
            if r[i] <= 0:
                return -np.inf
            if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
                return -np.inf
            if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
                return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                    return -np.inf
        return float(np.sum(r))

    def smooth_objective(z, w_fixed=None):
        """
        Objective for optimizer. z = [x0..xN, y0..yN, r0..rN, w]
        Uses smooth penalty for constraints.
        """
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N] if w_fixed is None else w_fixed
        h = 2.0 - w

        if w < 0.15 or h < 0.15 or np.any(r < 0.001):
            return 1e8

        # Primary objective: maximize sum of radii
        obj = -np.sum(r)

        # Smooth penalties
        penalty = 0.0
        # Wall constraints
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > w - r[i]:
                penalty += (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > h - r[i]:
                penalty += (y[i] - (h - r[i])) ** 2

        # Pairwise overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    penalty += overlap ** 2

        obj += 500.0 * penalty
        return obj

    def optimize_positions(x0, y0, r0, w0, time_budget=15.0):
        """Optimize positions and radii given initial guess."""
        z0 = np.concatenate([x0, y0, r0, [w0]])
        bounds = []
        for i in range(N):
            bounds.append((0.001, 2.0))  # x
        for i in range(N):
            bounds.append((0.001, 2.0))  # y
        for i in range(N):
            bounds.append((0.001, 0.5))  # r
        bounds.append((0.3, 1.7))  # w

        try:
            res = minimize(smooth_objective, z0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 50})
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            val = compute_value(x_opt, y_opt, r_opt, w_opt)
            if val > 0:
                return x_opt, y_opt, r_opt, w_opt, val
        except Exception:
            pass
        return None

    def grow_radii(x, y, r, w, time_budget=10.0):
        """Iteratively try to grow each radius slightly."""
        h = 2.0 - w
        improved = True
        while improved and (time.time() - start_time) < deadline - 10:
            improved = False
            for i in range(N):
                # Try growing radius i
                for _ in range(20):
                    r_new = r.copy()
                    r_new[i] *= 1.001
                    # Check if still valid after re-optimizing positions
                    z0 = np.concatenate([x, y, r_new, [w]])
                    bounds = []
                    for k in range(N):
                        bounds.append((0.001, 2.0))
                    for k in range(N):
                        bounds.append((0.001, 2.0))
                    for k in range(N):
                        bounds.append((r_new[k] * 0.95, r_new[k] * 1.05))
                    bounds.append((w * 0.95, w * 1.05))
                    try:
                        res = minimize(smooth_objective, z0, method='L-BFGS-B',
                                       bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
                        x_t = res.x[0:N]
                        y_t = res.x[N:2*N]
                        r_t = res.x[2*N:3*N]
                        w_t = res.x[3*N]
                        val = compute_value(x_t, y_t, r_t, w_t)
                        if val > 0:
                            x, y, r, w = x_t, y_t, r_t, w_t
                            improved = True
                        else:
                            break
                    except Exception:
                        break
        return x, y, r, w

    def hex_lattice_init(w0, n_circles=21):
        """Generate a hexagonal lattice initial guess."""
        h0 = 2.0 - w0
        # Hexagonal packing: rows offset
        # Estimate radius from area
        area = w0 * h0
        # 21 circles, hex packing efficiency ~0.9069
        r_est = np.sqrt(area / (21 * np.pi / 0.9069))
        r_est = min(r_est, 0.15)

        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)

        # Place in hex pattern
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        row = 0
        col = 0
        for i in range(n_circles):
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1
                col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[i] = cx
            y[i] = cy
            r[i] = r_est
            col += 1

        # Fill remaining circles in gaps
        for i in range(len(x)):
            if x[i] == 0 and y[i] == 0 and r[i] == 0:
                # Place at center of available space
                x[i] = w0 / 2
                y[i] = h0 / 2
                r[i] = r_est * 0.8

        return x, y, r, w0

    def random_init(w0, rng):
        """Random initial guess."""
        h0 = 2.0 - w0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            r[i] = rng.uniform(0.04, 0.1)
            x[i] = rng.uniform(r[i], w0 - r[i])
            y[i] = rng.uniform(r[i], h0 - r[i])
        return x, y, r, w0

    best_value = 0.0
    best_result = None

    rng = np.random.default_rng(12345)

    # Phase 1: Try multiple aspect ratios with hex lattice
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 2.0, 0.5, 1.1, 0.9, 1.3]
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        x0, y0, r0, w_init = hex_lattice_init(w0)
        result = optimize_positions(x0, y0, r0, w_init, time_budget=min(12, deadline - (time.time() - start_time) - 15))
        if result is not None:
            x_opt, y_opt, r_opt, w_opt, val = result
            if val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)

    # Phase 2: Random initializations
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        w0 = rng.uniform(0.6, 1.4)
        x0, y0, r0, w_init = random_init(w0, rng)
        result = optimize_positions(x0, y0, r0, w_init, time_budget=min(12, deadline - (time.time() - start_time) - 15))
        if result is not None:
            x_opt, y_opt, r_opt, w_opt, val = result
            if val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)

    # Phase 3: Grow radii from best result
    if best_result is not None:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = grow_radii(x_cur, y_cur, r_cur, w_cur, time_budget=min(20, deadline - (time.time() - start_time) - 10))
        val = compute_value(x_cur, y_cur, r_cur, w_cur)
        if val > best_value:
            best_value = val
            best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 4: Final polish with fine-tuning
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        # Multiple small perturbations
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x_p = x_cur + rng.normal(0, 0.005, N)
            y_p = y_cur + rng.normal(0, 0.005, N)
            r_p = r_cur * (1 + rng.normal(0, 0.002, N))
            r_p = np.clip(r_p, 0.001, 0.5)
            w_p = w_cur + rng.normal(0, 0.01)
            w_p = np.clip(w_p, 0.3, 1.7)
            result = optimize_positions(x_p, y_p, r_p, w_p, time_budget=min(8, deadline - (time.time() - start_time) - 10))
            if result is not None:
                x_opt, y_opt, r_opt, w_opt, val = result
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)

    if best_result is None:
        # Fallback: simple grid
        cols = 5
        rows = 5
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) / cols
            cy = (row + 0.5) / rows
            r0 = min(1.0 / cols, 1.0 / rows) * 0.45
            cells.append((cx, cy, r0))
        return np.array(cells)

    x_final, y_final, r_final, w_final = best_result
    # Safety shrink
    for _ in range(3):
        r_trial = r_final * (1 - 1e-9)
        if compute_value(x_final, y_final, r_trial, w_final) > 0:
            r_final = r_trial
        else:
            break

    return np.column_stack((x_final, y_final, r_final))
