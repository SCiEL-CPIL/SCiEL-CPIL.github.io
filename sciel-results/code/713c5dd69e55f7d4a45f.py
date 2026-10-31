"""Circle packing in perimeter-4 rectangle: unified SLSQP with free aspect ratio, force-directed repair, and iterative polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w
    penalty_weight = 800.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                dist_req = r[i] + r[j]
                if d < dist_req:
                    obj += penalty_weight * (dist_req - d) ** 2

        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

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

        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []

        # Grid-based starts with varying aspect ratios
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

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(20):
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
        if time.time() > deadline - 5:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=300)

        # SLSQP optimization
        maxiter = int(min(400, max(50, (deadline - (time.time() - start_time)) * 25)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: Iterative polish on best result
    if best_result is not None and time.time() < deadline - 10:
        polished = _polish(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # Phase 3: Another round of polish if time permits
    if best_result is not None and time.time() < deadline - 5:
        polished2 = _polish(best_result, deadline - (time.time() - start_time))
        if polished2 is not None:
            valid, val = check_validity(polished2)
            if valid and val > best_value:
                best_value = val
                best_result = polished2

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Safety shrink
    r = r * (1 - 1e-7)

    return np.column_stack((x, y, r))


def _force_relax(z, max_iter=300):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003

    for _ in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish(z, time_budget):
    """Polish a valid packing with force-directed radius growth and local optimization."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Try to grow radii iteratively while maintaining non-overlap
    growth_rate = 0.002
    max_grow_steps = 500

    for step in range(max_grow_steps):
        if time.time() > time_budget + (start_time if 'start_time' in dir() else time.time()):
            break

        # Attempt to grow all radii slightly
        r_new = r + growth_rate

        # Check if this growth maintains non-overlap
        valid = True
        for i in range(n):
            if x[i] - r_new[i] < -1e-9 or x[i] + r_new[i] > w + 1e-9:
                valid = False
                break
            if y[i] - r_new[i] < -1e-9 or y[i] + r_new[i] > h + 1e-9:
                valid = False
                break
            if valid:
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    r_sum = r_new[i] + r_new[j]
                    if d2 < r_sum ** 2 - 1e-9:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            # Try local optimization to adjust positions
            z_local = np.concatenate([x, y, r, [w]])
            maxiter = int(min(200, max(20, time_budget * 5)))
            try:
                res = minimize(objective, z_local, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
                z_opt = res.x
                valid, val = check_validity(z_opt)
                if valid:
                    x = z_opt[0::3][:n]
                    y = z_opt[1::3][:n]
                    r = z_opt[2::3][:n]
                    w = z_opt[-1]
                    h = 2.0 - w
                    # Retry growth after repositioning
                    continue
            except Exception:
                break
            break

        r = r_new

    # Final optimization
    z_final = np.concatenate([x, y, r, [w]])
    maxiter = int(min(300, max(50, time_budget * 10)))
    try:
        res = minimize(objective, z_final, method='SLSQP',
                       bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
        z_opt = res.x
        valid, val = check_validity(z_opt)
        if valid:
            return z_opt
    except Exception:
        pass

    return None
