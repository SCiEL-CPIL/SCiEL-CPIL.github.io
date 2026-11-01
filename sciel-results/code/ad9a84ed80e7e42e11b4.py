"""
Circle packing in perimeter-4 rectangle: 21 circles.
Combined approach:
  - Grid-based cellular decomposition for initialization (guarantees disjoint interiors)
  - Force-directed relaxation (fast, from inspiration)
  - SLSQP joint optimization over positions, radii, and width
  - Iterative radius growth with position relaxation
  - Nelder-Mead polish on positions
  - Multiple aspect-ratio restarts
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def is_valid(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def grid_init(w0, n=N):
        """
        Grid-based cellular decomposition initialization.
        Partition rectangle into a grid of cells, place one circle per cell
        with radius = min(cell_w, cell_h)/2 * safety_factor.
        This inherently guarantees disjoint interiors for the initial placement.
        """
        h0 = 2.0 - w0
        # Choose grid dimensions: try to make cells roughly square
        # For 21 circles, try various grid shapes
        best_val = -1
        best_x = best_y = best_r = None

        for rows in range(3, 8):
            for cols in range(3, 8):
                if rows * cols < n:
                    continue
                if rows * cols > n + 6:  # don't waste too many cells
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                # Inradius of each cell
                r_cell = min(cell_w, cell_h) / 2.0 * 0.95

                if r_cell < 0.02:
                    continue

                # Place circles in first 'n' cells of the grid
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
                    # Fill remaining with smaller circles at center
                    for i in range(idx, n):
                        x[i] = w0 / 2.0
                        y[i] = h0 / 2.0
                        r[i] = r_cell * 0.6

                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()

        if best_x is None:
            # Fallback: simple uniform grid
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

    def hex_init(w0, n=N):
        """Hexagonal lattice initial positions."""
        h0 = 2.0 - w0
        area = w0 * h0
        r_est = np.sqrt(area / (n * np.pi / 0.9069))
        r_est = min(r_est, 0.14)

        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, r_est)

        dx = 2.0 * r_est * 1.001
        dy = np.sqrt(3) * r_est * 1.001
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
            r[i] = r_est * 0.7

        return x, y, r

    def force_relax(x, y, r, w, max_iter=500, tol=1e-9):
        """Force-directed relaxation to resolve overlaps and wall violations."""
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.005
        min_dt = 0.0001

        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 100
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 100

            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    r_sum = r[i] + r[j]
                    overlap = r_sum - d
                    if overlap > 0:
                        ux = dx / d
                        uy = dy / d
                        f = overlap * 15.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f
                        max_overlap = max(max_overlap, overlap)

            # Check wall violations for max_overlap
            for i in range(N):
                if x[i] < r[i]:
                    max_overlap = max(max_overlap, r[i] - x[i])
                if x[i] > w - r[i]:
                    max_overlap = max(max_overlap, x[i] - (w - r[i]))
                if y[i] < r[i]:
                    max_overlap = max(max_overlap, r[i] - y[i])
                if y[i] > h - r[i]:
                    max_overlap = max(max_overlap, y[i] - (h - r[i]))

            if max_overlap < tol:
                break

            force_mag = np.max(np.sqrt(fx * fx + fy * fy))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.1:
                dt = min(0.01, dt * 1.1)

            x += dt * fx
            y += dt * fy

            # Clamp
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        return x, y

    def relax_positions(x, y, r, w, max_iter=200, ftol=1e-12):
        """Relax positions via L-BFGS-B to minimize overlap and wall penalties."""
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2 * N]
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
                        pen += ov ** 2
            return pen

        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]

        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 40})
            return res.x[:N], res.x[N:2 * N]
        except Exception:
            return x.copy(), y.copy()

    def try_grow(x, y, r, w):
        """Iteratively grow radii and relax positions."""
        h = 2.0 - w
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 5:
                    break
                for _ in range(30):
                    r_new = r.copy()
                    r_new[i] *= 1.002
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=50, ftol=1e-10)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
        return x, y, r, w

    def polish(x, y, r, w):
        """Nelder-Mead polish on positions only."""
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2 * N]
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
                        pen += ov ** 2
            return pen

        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            x_t, y_t = res.x[:N], res.x[N:2 * N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w

    def slsqp_optimize(x_init, y_init, r_init, w_init, time_limit=20.0):
        """SLSQP joint optimization over positions, radii, and width."""
        def objective(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w

            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6

            obj = -np.sum(r)

            # Overlap penalty
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    overlap = max(r[i] + r[j] - d, 0.0)
                    obj += 1000.0 * overlap * overlap

            # Boundary penalty
            for i in range(N):
                obj += 1000.0 * max(r[i] - x[i], 0.0) ** 2
                obj += 1000.0 * max(x[i] + r[i] - w, 0.0) ** 2
                obj += 1000.0 * max(r[i] - y[i], 0.0) ** 2
                obj += 1000.0 * max(y[i] + r[i] - h, 0.0) ** 2

            return obj

        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x_init
        z[1::3][:N] = y_init
        z[2::3][:N] = r_init
        z[-1] = w_init

        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            return None

        maxiter = int(min(300, max(50, remaining * 20)))
        try:
            res = minimize(objective, z, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})

            x_opt = res.x[0::3][:N]
            y_opt = res.x[1::3][:N]
            r_opt = res.x[2::3][:N]
            w_opt = res.x[-1]

            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt
        except Exception:
            pass

        return None

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: Try multiple aspect ratios with both grid and hex initialization
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try grid init
        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

        # Try hex init
        x0h, y0h, r0h = hex_init(w0)
        x_th, y_th = force_relax(x0h, y0h, r0h, w0, max_iter=300)
        if is_valid(x_th, y_th, r0h, w0):
            val_h = sum_radii(r0h)
            if val_h > best_value:
                best_value = val_h
                best_result = (x_th.copy(), y_th.copy(), r0h.copy(), w0)

    # Phase 2: SLSQP joint optimization from best
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result
        result = slsqp_optimize(x_cur, y_cur, r_cur, w_cur, time_limit=25.0)
        if result is not None:
            x_s, y_s, r_s, w_s = result
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 3: Grow radii from best
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 4: Polish
    if best_result is not None and time.time() < deadline - 10:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 5: Additional random restarts if time permits
    if time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue

            # Alternate between grid and hex
            if rng.random() < 0.5:
                x0, y0, r0 = grid_init(w0)
            else:
                x0, y0, r0 = hex_init(w0)

            # Perturb
            x0 += rng.normal(0, 0.01, N)
            y0 += rng.normal(0, 0.01, N)

            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=200)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)

            if val > best_value * 0.95:  # Promising
                # Try SLSQP
                if time.time() < deadline - 12:
                    result = slsqp_optimize(x_t, y_t, r0, w0, time_limit=10.0)
                    if result is not None:
                        x_s, y_s, r_s, w_s = result
                        val_s = sum_radii(r_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
                            continue

                # Try grow
                if time.time() < deadline - 8:
                    x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0)
                    if is_valid(x_g, y_g, r_g, w_g):
                        val_g = sum_radii(r_g)
                        if val_g > best_value:
                            best_value = val_g
                            best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g)

    # Phase 6: Final polish
    if best_result is not None and time.time() < deadline - 5:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    if best_result is None:
        # Fallback: simple grid
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
    for _ in range(5):
        r_trial = r_final * (1 - 1e-10)
        if is_valid(x_final, y_final, r_trial, w_final):
            r_final = r_trial
        else:
            break

    return np.column_stack((x_final, y_final, r_final))
