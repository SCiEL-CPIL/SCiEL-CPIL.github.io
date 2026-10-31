"""Circle packing in perimeter-4 rectangle: SLSQP with hard constraints + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)

    def is_valid_fast(x, y, r, w, h, tol=1e-8):
        """Fast validity check."""
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

    def expand_radii(x, y, r, w, h, time_budget=5.0):
        """Try to expand radii greedily."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()

        for _ in range(50):
            if time.time() - start > time_budget:
                break

            improved = False
            order = np.argsort(r)
            for i in order:
                if time.time() - start > time_budget:
                    break

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

                r_max = max(0.0, r_max)

                if r_max > r[i] + 1e-8:
                    lo = r[i]
                    hi = r_max
                    for _ in range(20):
                        mid = (lo + hi) / 2.0
                        r_test = r.copy()
                        r_test[i] = mid
                        if is_valid_fast(x, y, r_test, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-9:
                        r[i] = lo
                        improved = True

            if not improved:
                break

        return x, y, r

    def local_position_search(x, y, r, w, h, time_budget=10.0):
        """Local search: try moving each circle to find better positions."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()

        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        steps = [0.1, 0.05, 0.03, 0.02, 0.01, 0.005]

        for rnd in range(20):
            if time.time() - start > time_budget:
                break

            improved = False
            for i in range(N):
                if time.time() - start > time_budget:
                    break

                best_move = None
                best_gain = 1e-6

                for angle in angles:
                    for step in steps:
                        nx = x[i] + step * np.cos(angle)
                        ny = y[i] + step * np.sin(angle)

                        if nx < r[i] or nx > w - r[i]:
                            continue
                        if ny < r[i] or ny > h - r[i]:
                            continue

                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            d2 = dx * dx + dy * dy
                            rs = r[i] + r[j]
                            if d2 < rs * rs - 1e-9:
                                valid = False
                                break

                        if not valid:
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

    def slsqp_optimize(x0, y0, r0, w0, h0, time_budget=15.0):
        """
        SLSQP optimization with hard constraints.
        Variables: [x_0, ..., x_{N-1}, y_0, ..., y_{N-1}, r_0, ..., r_{N-1}, w]
        """
        n_vars = 3 * N + 1  # x, y, r for each circle + w

        def unpack(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            w = v[3*N]
            return x, y, r, w

        def objective(v):
            x, y, r, w = unpack(v)
            return -np.sum(r)

        # Build constraints
        constraints = []
        bounds = []

        # Bounds for x, y, r, w
        for i in range(N):
            bounds.append((0.0, 2.0))  # x_i
        for i in range(N):
            bounds.append((0.0, 2.0))  # y_i
        for i in range(N):
            bounds.append((0.001, 1.0))  # r_i
        bounds.append((0.3, 1.7))  # w

        # Wall constraints: x_i - r_i >= 0, w - x_i - r_i >= 0, y_i - r_i >= 0, h - y_i - r_i >= 0
        # where h = 2 - w
        for i in range(N):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[i] - v[2*N + i]
            })
            # w - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N + i]
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[N + i] - v[2*N + i]
            })
            # (2 - w) - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 2.0 - v[3*N] - v[N + i] - v[2*N + i]
            })

        # Circle-circle constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (v[2*N+i] + v[2*N+j])**2
            })

        # Initial point
        v0 = np.empty(n_vars)
        v0[:N] = x0
        v0[N:2*N] = y0
        v0[2*N:3*N] = r0
        v0[3*N] = w0

        maxiter = int(min(500, time_budget * 60))
        try:
            res = minimize(
                objective, v0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 100}
            )
            v_opt = res.x
            xo = v_opt[:N]
            yo = v_opt[N:2*N]
            ro = v_opt[2*N:3*N]
            wo = v_opt[3*N]
            ho = 2.0 - wo
            if is_valid_fast(xo, yo, ro, wo, ho):
                return xo, yo, ro, wo
        except Exception:
            pass
        return None

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

        # Random starts
        for seed_val in range(8):
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
        if time.time() > deadline - 30:
            break

        h0 = 2.0 - w0

        # Step 1: Expand radii greedily to get a feasible starting point
        x, y, r = expand_radii(x0, y0, r0, w0, h0, time_budget=3.0)

        # Check validity after expansion
        if is_valid_fast(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[:N] = x
                z[N:2*N] = y
                z[2*N:3*N] = r
                z[3*N] = w0
                best_result = z.copy()

        # Step 2: SLSQP with hard constraints
        if time.time() < deadline - 20:
            time_left = deadline - time.time() - 15
            result = slsqp_optimize(x, y, r, w0, h0, time_budget=min(12.0, time_left))
            if result is not None:
                xr, yr, rr, wr = result
                hr = 2.0 - wr
                if is_valid_fast(xr, yr, rr, wr, hr):
                    val_r = float(np.sum(rr))
                    if val_r > best_value:
                        best_value = val_r
                        z = np.empty(3 * N + 1)
                        z[:N] = xr
                        z[N:2*N] = yr
                        z[2*N:3*N] = rr
                        z[3*N] = wr
                        best_result = z.copy()

    # --- Phase 2: Polish best result ---
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[:N].copy()
        y = best_result[N:2*N].copy()
        r = best_result[2*N:3*N].copy()
        w = best_result[3*N]
        h = 2.0 - w

        # Expand radii
        x, y, r = expand_radii(x, y, r, w, h, time_budget=min(5.0, deadline - time.time() - 8))
        if is_valid_fast(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[:N] = x
                z[N:2*N] = y
                z[2*N:3*N] = r
                z[3*N] = w
                best_result = z.copy()

        # Local position search
        if time.time() < deadline - 8:
            x, y, r = local_position_search(x, y, r, w, h, time_budget=min(8.0, deadline - time.time() - 5))
            if is_valid_fast(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[:N] = x
                    z[N:2*N] = y
                    z[2*N:3*N] = r
                    z[3*N] = w
                    best_result = z.copy()

        # Final SLSQP polish with tighter tolerance
        if time.time() < deadline - 5:
            time_left = deadline - time.time() - 3
            result = slsqp_optimize(x, y, r, w, h, time_budget=min(10.0, time_left))
            if result is not None:
                xr, yr, rr, wr = result
                hr = 2.0 - wr
                if is_valid_fast(xr, yr, rr, wr, hr):
                    val_r = float(np.sum(rr))
                    if val_r > best_value:
                        best_value = val_r
                        z = np.empty(3 * N + 1)
                        z[:N] = xr
                        z[N:2*N] = yr
                        z[2*N:3*N] = rr
                        z[3*N] = wr
                        best_result = z.copy()

        # Final expansion
        if time.time() < deadline - 3:
            x = best_result[:N].copy()
            y = best_result[N:2*N].copy()
            r = best_result[2*N:3*N].copy()
            w = best_result[3*N]
            h = 2.0 - w
            x, y, r = expand_radii(x, y, r, w, h, time_budget=min(3.0, deadline - time.time() - 1))
            if is_valid_fast(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[:N] = x
                    z[N:2*N] = y
                    z[2*N:3*N] = r
                    z[3*N] = w
                    best_result = z.copy()

    # --- Phase 3: Try different aspect ratios for the best configuration ---
    if best_result is not None and time.time() < deadline - 8:
        x = best_result[:N].copy()
        y = best_result[N:2*N].copy()
        r = best_result[2*N:3*N].copy()
        w_orig = best_result[3*N]
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

            # Expand radii
            xs, ys, rs = expand_radii(xs, ys, rs, w_try, h_try, time_budget=2.0)

            if is_valid_fast(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[:N] = xs
                    z[N:2*N] = ys
                    z[2*N:3*N] = rs
                    z[3*N] = w_try
                    best_result = z.copy()

    if best_result is None:
        # Fallback: simple grid
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        return np.column_stack((x, y, r))

    # Final output
    x = best_result[:N]
    y = best_result[N:2*N]
    r = best_result[2*N:3*N]
    w = best_result[3*N]
    h = 2.0 - w

    # Final safety: tiny shrink if needed
    for scale in [1.0, 0.99999, 0.9999, 0.999]:
        r_trial = r * scale
        if is_valid_fast(x, y, r_trial, w, h, tol=1e-9):
            r = r_trial
            break

    return np.column_stack((x, y, r))
