"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a structural approach:
  1. Hexagonal lattice initialization with variable aspect ratio.
  2. L-BFGS-B optimization with smooth penalty for overlaps.
  3. Iterative radius growth via local optimization.
  4. Multiple restarts with different aspect ratios and lattice tilts.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, tol=1e-9):
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

    def objective(z, w_fixed=None):
        """
        Objective for L-BFGS-B.
        z = [x0..x20, y0..y20, r0..r20] (63 vars)
        If w_fixed is not None, w is fixed; otherwise w = z[63].
        """
        n_vars = 3 * N if w_fixed is None else 3 * N
        if w_fixed is None:
            w = z[-1]
            x = z[0:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
        else:
            w = w_fixed
            x = z[0:N]
            y = z[N:2*N]
            r = z[2*N:3*N]

        h = 2.0 - w
        if w < 0.2 or h < 0.2:
            return 1e6
        if np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty (smooth)
        penalty_weight = 500.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    obj += penalty_weight * overlap * overlap

        # Boundary penalty
        for i in range(N):
            obj += penalty_weight * max(r[i] - x[i], 0.0) ** 2
            obj += penalty_weight * max(x[i] + r[i] - w, 0.0) ** 2
            obj += penalty_weight * max(r[i] - y[i], 0.0) ** 2
            obj += penalty_weight * max(y[i] + r[i] - h, 0.0) ** 2

        return obj

    def objective_fixed_w(z, w):
        """Objective with fixed w for radius growth."""
        h = 2.0 - w
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]

        if h < 0.2:
            return 1e6
        if np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        penalty_weight = 1000.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    obj += penalty_weight * overlap * overlap

        for i in range(N):
            obj += penalty_weight * max(r[i] - x[i], 0.0) ** 2
            obj += penalty_weight * max(x[i] + r[i] - w, 0.0) ** 2
            obj += penalty_weight * max(r[i] - y[i], 0.0) ** 2
            obj += penalty_weight * max(y[i] + r[i] - h, 0.0) ** 2

        return obj

    def hex_lattice_init(w, h, tilt=0.0):
        """
        Generate hexagonal lattice positions for N circles in rectangle w x h.
        tilt: small rotation to break symmetry.
        """
        # Compute spacing for hexagonal packing
        # In hex packing, rows are offset by half spacing
        # Try to fit N circles
        # Start with a rough estimate
        area = w * h
        avg_radius = np.sqrt(area / (N * np.pi * 2.5))  # rough
        spacing = avg_radius * 2.2  # hex spacing

        # Try different numbers of columns
        best_x = None
        best_y = None
        best_score = -np.inf

        for n_cols in range(3, 8):
            n_rows = int(np.ceil(N / n_cols))
            if n_rows < 2:
                continue

            # Hexagonal arrangement
            row_spacing = spacing * np.sqrt(3) / 2
            col_spacing = spacing

            # Center the lattice
            x_total = (n_cols - 1) * col_spacing
            y_total = (n_rows - 1) * row_spacing

            x_start = (w - x_total) / 2
            y_start = (h - y_total) / 2

            x = np.zeros(N)
            y = np.zeros(N)

            for i in range(N):
                row, col = divmod(i, n_cols)
                if row >= n_rows:
                    break
                # Hex offset: odd rows shifted by half col_spacing
                cx = x_start + col * col_spacing
                if row % 2 == 1:
                    cx += col_spacing / 2
                cy = y_start + row * row_spacing

                # Apply tilt
                cx_t = cx * np.cos(tilt) - cy * np.sin(tilt)
                cy_t = cx * np.sin(tilt) + cy * np.cos(tilt)

                x[i] = cx_t
                y[i] = cy_t

            # Check if all circles fit
            valid = True
            for i in range(N):
                if x[i] < 0 or x[i] > w or y[i] < 0 or y[i] > h:
                    valid = False
                    break
            if not valid:
                continue

            # Score: how well they fit (lower spread is better)
            score = -np.var(x[:N]) - np.var(y[:N])
            if score > best_score:
                best_score = score
                best_x = x.copy()
                best_y = y.copy()

        if best_x is None:
            # Fallback: grid
            n_cols = 5
            n_rows = int(np.ceil(N / n_cols))
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, n_cols)
                x[i] = (col + 0.5) / n_cols * w
                y[i] = (row + 0.5) / n_rows * h
            best_x = x
            best_y = y

        # Initial radii
        r = np.full(N, min(w / 6, h / 4) * 0.45)

        return best_x, best_y, r

    def optimize_with_w(x0, y0, r0, w0, time_limit=20.0):
        """Optimize with variable w."""
        z = np.zeros(3 * N + 1)
        z[0:N] = x0
        z[N:2*N] = y0
        z[2*N:3*N] = r0
        z[-1] = w0

        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None

        maxiter = int(min(300, max(50, remaining * 20)))
        try:
            res = minimize(
                objective, z, method='L-BFGS-B',
                bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[-1]

            valid, val = check_validity(x_opt, y_opt, r_opt, w_opt)
            if valid:
                return x_opt, y_opt, r_opt, w_opt
        except Exception:
            pass
        return None

    def grow_radii(x, y, r, w, time_budget=15.0):
        """Iteratively grow radii by optimizing positions."""
        h = 2.0 - w
        for _ in range(50):
            if time.time() > start_time + time_budget:
                break

            z = np.zeros(3 * N)
            z[0:N] = x
            z[N:2*N] = y
            z[2*N:3*N] = r

            maxiter = int(min(100, max(20, (time_budget - (time.time() - start_time)) * 10)))
            try:
                res = minimize(
                    lambda z: objective_fixed_w(z, w), z, method='L-BFGS-B',
                    bounds=[(0, w)] * N + [(0, h)] * N + [(0, min(w, h) / 2)] * N,
                    options={'maxiter': maxiter, 'ftol': 1e-12}
                )
                x_new = res.x[0:N]
                y_new = res.x[N:2*N]
                r_new = res.x[2*N:3*N]

                valid, val = check_validity(x_new, y_new, r_new, w)
                if valid and val > np.sum(r):
                    x, y, r = x_new, y_new, r_new
                else:
                    break
            except Exception:
                break

        return x, y, r

    # Generate diverse initial guesses
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 2.0, 0.5, 0.9, 1.1, 1.3]
    tilts = [0.0, 0.05, -0.05, 0.1, -0.1]

    guesses = []
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        for tilt in tilts:
            x0, y0, r0 = hex_lattice_init(w0, h0, tilt)
            guesses.append((x0, y0, r0, w0))

    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        x0 = rng.uniform(0.1, w0 - 0.1, N)
        y0 = rng.uniform(0.1, h0 - 0.1, N)
        r0 = np.full(N, rng.uniform(0.05, 0.12))
        guesses.append((x0, y0, r0, w0))

    # Optimize each initial guess
    for i, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break

        result = optimize_with_w(x0, y0, r0, w0, time_limit=min(20, remaining))

        if result is not None:
            x_opt, y_opt, r_opt, w_opt = result
            valid, val = check_validity(x_opt, y_opt, r_opt, w_opt)
            if valid and val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)

    # Phase 2: Polish best result with radius growth
    if best_result is not None and time.time() < deadline - 40:
        x_cur, y_cur, r_cur, w_cur = best_result

        # Grow radii
        x_grow, y_grow, r_grow = grow_radii(x_cur, y_cur, r_cur, w_cur,
                                             time_budget=deadline - (time.time() - start_time) - 20)
        valid, val = check_validity(x_grow, y_grow, r_grow, w_cur)
        if valid and val > best_value:
            best_value = val
            x_cur, y_cur, r_cur = x_grow, y_grow, r_grow

        # Final polish with variable w
        if time.time() < deadline - 15:
            z = np.zeros(3 * N + 1)
            z[0:N] = x_cur
            z[N:2*N] = y_cur
            z[2*N:3*N] = r_cur
            z[-1] = w_cur

            remaining = deadline - (time.time() - start_time) - 5
            if remaining > 5:
                maxiter = int(min(200, max(50, remaining * 15)))
                try:
                    res = minimize(
                        objective, z, method='L-BFGS-B',
                        bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                        options={'maxiter': maxiter, 'ftol': 1e-14}
                    )
                    x_opt = res.x[0:N]
                    y_opt = res.x[N:2*N]
                    r_opt = res.x[2*N:3*N]
                    w_opt = res.x[-1]
                    valid, val = check_validity(x_opt, y_opt, r_opt, w_opt)
                    if valid and val > best_value:
                        best_value = val
                        x_cur, y_cur, r_cur, w_cur = x_opt, y_opt, r_opt, w_opt
                except Exception:
                    pass

    if best_result is None:
        # Fallback: simple grid in unit square
        cols = 5
        rows = 5
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) / cols
            cy = (row + 0.5) / rows
            r0 = min(1.0 / cols, 1.0 / rows) * 0.48
            cells.append((cx, cy, r0))
        return np.array(cells)

    # Final safety check with minimal shrink
    x_final, y_final, r_final, w_final = x_cur, y_cur, r_cur, w_cur
    for _ in range(5):
        r_trial = r_final * (1 - 1e-9)
        if check_validity(x_final, y_final, r_trial, w_final)[0]:
            r_final = r_trial
        else:
            break

    return np.column_stack((x_final, y_final, r_final))
