import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 145  # 5s margin before 150s limit

    # --- Helper: compute value (sum of radii) with validation ---
    def compute_value(x, w, h):
        """
        x: shape (3N,) -> [x0, y0, r0, x1, y1, r1, ...]
        Returns sum of radii if valid, else -inf
        """
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]

        if np.any(rs < 1e-9):
            return -np.inf
        if np.any(xs - rs < -1e-8) or np.any(xs + rs > w + 1e-8):
            return -np.inf
        if np.any(ys - rs < -1e-8) or np.any(ys + rs > h + 1e-8):
            return -np.inf

        # Pairwise distance check
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if dist < rs[i] + rs[j] - 1e-8:
                    return -np.inf

        return np.sum(rs)

    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def update_best(x, w, h):
        nonlocal best_value, best_result, best_w, best_h
        val = compute_value(x, w, h)
        if val > best_value:
            best_value = val
            best_result = x.copy()
            best_w = w
            best_h = h

    # --- Objective for SLSQP (fixed w, h) ---
    def make_objective(w, h):
        def objective(x):
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]

            penalty = 0.0

            # Boundary penalties
            for i in range(N):
                if xs[i] < rs[i]:
                    penalty += (rs[i] - xs[i])**2
                if xs[i] > w - rs[i]:
                    penalty += (xs[i] - (w - rs[i]))**2
                if ys[i] < rs[i]:
                    penalty += (rs[i] - ys[i])**2
                if ys[i] > h - rs[i]:
                    penalty += (ys[i] - (h - rs[i]))**2

            # Overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2

            return -np.sum(rs) + penalty

        return objective

    # --- Initial configuration generator ---
    def make_initial(w, h, aspect):
        """
        Create an initial configuration.
        aspect: w/h ratio
        """
        x0 = np.zeros(3 * N)

        # Grid-based initial guess
        # For 21 circles, a 5x4 grid (20 cells) + 1 extra is reasonable
        nx = 5
        ny = 4

        idx = 0
        for i in range(nx):
            for j in range(ny):
                if idx >= N:
                    break
                x0[3*idx] = (i + 0.5) * w / nx
                x0[3*idx + 1] = (j + 0.5) * h / ny
                # Initial radius: fit in cell
                cell_w = w / nx
                cell_h = h / ny
                r_init = min(cell_w, cell_h) * 0.45
                x0[3*idx + 2] = r_init
                idx += 1

        # Fill remaining circles (should be 1 for 5x4 grid)
        while idx < N:
            x0[3*idx] = w / 2
            x0[3*idx + 1] = h / 2
            x0[3*idx + 2] = min(w, h) * 0.1
            idx += 1

        return x0

    # --- Simulated Annealing for aspect ratio + positions ---
    def simulated_annealing():
        nonlocal best_value, best_result, best_w, best_h
        start_time = time.time()
        deadline = start_time + 140  # Leave 5s margin

        # Initial aspect ratio
        aspect = 1.0
        w = 2.0 * aspect / (aspect + 1)
        h = 2.0 / (aspect + 1)

        x = make_initial(w, h, aspect)
        update_best(x, w, h)

        T = 0.05  # Initial temperature
        cooling = 0.995
        max_iter = 2000

        for it in range(max_iter):
            if time.time() - start_time > deadline:
                break

            # Perturb aspect ratio
            new_aspect = aspect + np.random.normal(0, 0.1)
            new_aspect = np.clip(new_aspect, 0.3, 3.0)
            new_w = 2.0 * new_aspect / (new_aspect + 1)
            new_h = 2.0 / (new_aspect + 1)

            # Perturb positions and radii
            new_x = x.copy()
            for i in range(N):
                new_x[3*i] += np.random.normal(0, 0.02)
                new_x[3*i + 1] += np.random.normal(0, 0.02)
                new_x[3*i + 2] += np.random.normal(0, 0.01)

            # Clip to valid ranges
            for i in range(N):
                new_x[3*i] = np.clip(new_x[3*i], 0, new_w)
                new_x[3*i + 1] = np.clip(new_x[3*i + 1], 0, new_h)
                new_x[3*i + 2] = np.clip(new_x[3*i + 2], 0.01, min(new_w, new_h)/2)

            old_val = compute_value(x, w, h)
            new_val = compute_value(new_x, new_w, new_h)

            if new_val == -np.inf:
                # Reject invalid
                if np.random.random() > np.exp((new_val - old_val) / T) if old_val != -np.inf else False:
                    x = new_x
                    w = new_w
                    h = new_h
                    aspect = new_aspect
            elif new_val > old_val or np.random.random() < np.exp((new_val - old_val) / T):
                x = new_x
                w = new_w
                h = new_h
                aspect = new_aspect
                update_best(x, w, h)

            T *= cooling

            # Periodically refine with SLSQP
            if it % 50 == 0 and time.time() - start_time < deadline - 5:
                try:
                    obj = make_objective(w, h)
                    bounds = []
                    for i in range(N):
                        bounds.append((0, w))
                        bounds.append((0, h))
                        bounds.append((0.01, min(w, h)/2))

                    res = minimize(obj, x, method='SLSQP', bounds=bounds, options={'maxiter': 30, 'ftol': 1e-8})
                    if res.fun < obj(x):
                        x = res.x
                        update_best(x, w, h)
                except Exception:
                    pass

    # --- Multi-start SLSQP with different aspect ratios ---
    def multi_start_slsqp():
        nonlocal best_value, best_result, best_w, best_h
        start_time = time.time()
        deadline = start_time + 140

        aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.667, 2.0, 0.5, 1.333, 0.75, 1.1, 0.9]

        for ar in aspect_ratios:
            if time.time() - start_time > deadline:
                break

            w = 2.0 * ar / (ar + 1)
            h = 2.0 / (ar + 1)

            x0 = make_initial(w, h, ar)
            update_best(x0, w, h)

            try:
                obj = make_objective(w, h)
                bounds = []
                for i in range(N):
                    bounds.append((0, w))
                    bounds.append((0, h))
                    bounds.append((0.01, min(w, h)/2))

                res = minimize(obj, x0, method='SLSQP', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
                if res.fun < obj(x0):
                    update_best(res.x, w, h)
            except Exception:
                pass

    # --- Run optimization strategies ---
    # First, multi-start SLSQP (fast, deterministic)
    multi_start_slsqp()

    # Then, simulated annealing (stochastic, explores more)
    if time.time() - start_time < time_limit - 10:
        simulated_annealing()

    # --- Final refinement with SLSQP on best found ---
    if best_result is not None and time.time() - start_time < time_limit - 5:
        try:
            obj = make_objective(best_w, best_h)
            bounds = []
            for i in range(N):
                bounds.append((0, best_w))
                bounds.append((0, best_h))
                bounds.append((0.01, min(best_w, best_h)/2))

            res = minimize(obj, best_result, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < obj(best_result):
                update_best(res.x, best_w, best_h)
        except Exception:
            pass

    # --- Format result ---
    if best_result is not None:
        result = np.zeros((N, 3))
        result[:, 0] = best_result[0::3]
        result[:, 1] = best_result[1::3]
        result[:, 2] = best_result[2::3]

        # Apply small epsilon to ensure strict validity
        for i in range(N):
            result[i, 0] = np.clip(result[i, 0], result[i, 2] + 1e-6, best_w - result[i, 2] - 1e-6)
            result[i, 1] = np.clip(result[i, 1], result[i, 2] + 1e-6, best_h - result[i, 2] - 1e-6)
            result[i, 2] -= 1e-5

        # Final validation
        val = compute_value(result[:, 0], result[:, 1], best_w, best_h) if False else None
        # Re-validate with the adjusted values
        xs = result[:, 0]
        ys = result[:, 1]
        rs = result[:, 2]

        valid = True
        if np.any(rs < 1e-9):
            valid = False
        if np.any(xs - rs < -1e-8) or np.any(xs + rs > best_w + 1e-8):
            valid = False
        if np.any(ys - rs < -1e-8) or np.any(ys + rs > best_h + 1e-8):
            valid = False

        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if dist < rs[i] + rs[j] - 1e-8:
                    valid = False
                    break

        if not valid:
            # Fallback to seed-like grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

        return result
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
