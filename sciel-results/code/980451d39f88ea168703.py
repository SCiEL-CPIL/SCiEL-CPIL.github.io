import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin

    best_value = 0.0
    best_result = None

    def is_valid(x, y, r, w, h, tol=1e-8):
        if w < 0.1 or h < 0.1:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                rs = r[i] + r[j]
                if d2 < rs * rs - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def expand_all(x, y, r, w, h, max_rounds=100):
        """Iteratively expand each circle to its maximum possible radius."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        for rnd in range(max_rounds):
            improved = False
            for i in range(n):
                # Max radius from walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                # Max radius from other circles
                dx = x[i] - x
                dy = y[i] - y
                d = np.sqrt(dx * dx + dy * dy)
                d[i] = np.inf
                r_allowed = d - r
                r_max = min(r_wall, np.min(r_allowed))
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def full_slsqp(x0, y0, r0, w0, h0, time_budget=30):
        """SLSQP with explicit constraints for all pairwise and wall constraints."""
        n = N
        # Variables: x(0..n-1), y(n..2n-1), r(2n..3n-1), w(3n)
        z0 = np.zeros(3 * n + 1)
        z0[:n] = x0
        z0[n:2*n] = y0
        z0[2*n:3*n] = r0
        z0[3*n] = w0

        h_val = 2.0 - w0

        def objective(z):
            return -np.sum(z[2*n:3*n])  # Maximize sum of radii

        # Build constraints
        constraints = []
        # Wall constraints: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
        for i in range(n):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[i] - z[2*n + i],
                'jac': lambda z, i=i: np.concatenate([np.eye(1, 3*n+1)[i], np.zeros(3*n)])
            })
            # w - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[3*n] - z[i] - z[2*n + i],
                'jac': lambda z, i=i: np.concatenate([np.eye(1, 3*n+1)[i] * -1, np.zeros(3*n)])
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: z[n + i] - z[2*n + i],
                'jac': lambda z, i=i: np.concatenate([np.eye(1, 3*n+1)[n + i], np.zeros(3*n)])
            })
            # h - y_i - r_i >= 0 => 2 - w - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i: 2.0 - z[3*n] - z[n + i] - z[2*n + i],
                'jac': lambda z, i=i: np.concatenate([np.eye(1, 3*n+1)[n + i] * -1, np.zeros(3*n)])
            })

        # Pairwise constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Linearized: (x_i - x_j) * u_x + (y_i - y_j) * u_y >= r_i + r_j - d
        # where u is the unit vector from j to i, d is current distance
        # For SLSQP, we use the constraint: d_ij - (r_i + r_j) >= 0
        # This is nonlinear but SLSQP can handle it
        for i in range(n):
            for j in range(i + 1, n):
                def pair_fun(z, i=i, j=j):
                    dx = z[i] - z[j]
                    dy = z[n + i] - z[n + j]
                    d = np.sqrt(dx * dx + dy * dy)
                    return d - z[2*n + i] - z[2*n + j]

                def pair_jac(z, i=i, j=j):
                    dx = z[i] - z[j]
                    dy = z[n + i] - z[n + j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-10:
                        d = 1e-10
                    ux = dx / d
                    uy = dy / d
                    jac = np.zeros(3 * n + 1)
                    jac[i] = ux
                    jac[j] = -ux
                    jac[n + i] = uy
                    jac[n + j] = -uy
                    jac[2*n + i] = -1
                    jac[2*n + j] = -1
                    return jac

                constraints.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})

        bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

        maxiter = int(min(200, time_budget * 15))
        try:
            res = minimize(objective, z0, method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-8, 'maxls': 50})
            z_opt = res.x
            x = z_opt[:n]
            y = z_opt[n:2*n]
            r = z_opt[2*n:3*n]
            w = z_opt[3*n]
            h = 2.0 - w
            valid, val = is_valid(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
            # Try slight shrink
            r *= 0.98
            valid, val = is_valid(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
            return None
        except Exception:
            return None

    def simulated_annealing(x, y, r, w, h, time_budget=10):
        """SA to perturb positions and escape local optima."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        rng = np.random.default_rng(42)

        # Current state
        cur_x, cur_y, cur_r = x.copy(), y.copy(), r.copy()
        cur_val = 0.0
        valid, cur_val = is_valid(cur_x, cur_y, cur_r, w, h)
        if not valid:
            return x, y, r

        best_x, best_y, best_r = cur_x.copy(), cur_y.copy(), cur_r.copy()
        best_val = cur_val

        T_start = 0.05
        T_end = 0.001
        end_time = time.time() + time_budget
        step = 0

        while time.time() < end_time:
            T = T_start * (T_end / T_start) ** (step / max(1, int(time_budget * 100)))
            T = max(T, T_end)

            # Pick a random circle
            i = rng.integers(n)
            # Random perturbation
            dx = rng.normal(0, T)
            dy = rng.normal(0, T)
            new_x = cur_x[i] + dx
            new_y = cur_y[i] + dy

            # Clamp to walls
            new_x = np.clip(new_x, cur_r[i], w - cur_r[i])
            new_y = np.clip(new_y, cur_r[i], h - cur_r[i])

            # Check validity of new position
            valid_move = True
            for j in range(n):
                if j == i:
                    continue
                ddx = new_x - cur_x[j]
                ddy = new_y - cur_y[j]
                d2 = ddx * ddx + ddy * ddy
                rs = cur_r[i] + cur_r[j]
                if d2 < rs * rs - 1e-8:
                    valid_move = False
                    break

            if valid_move:
                cur_x[i] = new_x
                cur_y[i] = new_y
                # Try to expand all radii
                new_x, new_y, new_r = expand_all(cur_x, cur_y, cur_r, w, h, max_rounds=20)
                valid, new_val = is_valid(new_x, new_y, new_r, w, h)
                if valid and new_val > cur_val:
                    cur_val = new_val
                    cur_r = new_r
                    if new_val > best_val:
                        best_val = new_val
                        best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()
            else:
                # Accept with probability exp(-delta/T) if it's worse
                # But we only accept valid moves, so skip
                pass

            step += 1

        return best_x, best_y, best_r

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 2.0, 0.5, 1.3, 0.75]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(10):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.035, 0.11)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()
    guesses.sort(key=lambda g: abs(g[3] - 1.0))

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Phase 1: Expand initial radii
        x, y, r = expand_all(x0, y0, r0, w0, h0, max_rounds=50)

        # Phase 2: Full SLSQP with constraints
        remaining = deadline - (time.time() - start_time)
        time_budget = min(25, remaining - 15)
        if time_budget > 5:
            result = full_slsqp(x, y, r, w0, h0, time_budget)
            if result is not None:
                x, y, r, w, h, val = result
            else:
                valid, val = is_valid(x, y, r, w0, h0)
                if not valid:
                    continue
                w, h = w0, h0
        else:
            valid, val = is_valid(x, y, r, w0, h0)
            if not valid:
                continue
            w, h = w0, h0

        # Phase 3: Simulated Annealing
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            x, y, r = simulated_annealing(x, y, r, w, h, time_budget=min(8, remaining - 10))
            # Expand again
            x, y, r = expand_all(x, y, r, w, h, max_rounds=50)

        # Final validity check
        valid, val = is_valid(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Final polish
    if best_result is not None and time.time() < deadline - 5:
        x, y, r, w, h = best_result
        x, y, r = expand_all(x, y, r, w, h, max_rounds=100)
        valid, val = is_valid(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
