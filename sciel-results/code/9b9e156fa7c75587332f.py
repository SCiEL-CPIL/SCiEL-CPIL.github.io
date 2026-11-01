"""Circle packing in perimeter-4 rectangle: Annealed momentum + SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check all non-overlap and boundary constraints."""
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


def _compute_energy(x, y, r, w):
    """Compute penalty energy (0 if valid)."""
    h = 2.0 - w
    if w < 0.15 or h < 0.15 or np.any(r < 0):
        return 1e6
    energy = 0.0
    energy += np.sum(np.maximum(r - x, 0.0) ** 2)
    energy += np.sum(np.maximum(x + r - w, 0.0) ** 2)
    energy += np.sum(np.maximum(r - y, 0.0) ** 2)
    energy += np.sum(np.maximum(y + r - h, 0.0) ** 2)
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    overlap = np.maximum(r[iu[0]] + r[iu[1]] - d, 0.0)
    energy += np.sum(overlap ** 2)
    return energy


def _anneal_momentum(z, time_budget, seed=0):
    """Simulated annealing with momentum and radial growth."""
    rng = np.random.default_rng(seed)
    x = z[0::3][:N].copy().astype(float)
    y = z[1::3][:N].copy().astype(float)
    r = z[2::3][:N].copy().astype(float)
    w = z[-1].copy()
    h = 2.0 - w

    # Force relaxation to get valid starting point
    x, y, r, w = _force_relax(x, y, r, w, max_iter=300)
    if not _is_valid(x, y, r, w, 2.0 - w):
        return None

    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w.copy()
    best_val = float(np.sum(r))

    T0 = 0.08
    T_min = 0.0003
    n_steps = 12000
    # Precompute aspect ratio schedule (anneal w as well)
    w_schedule = np.linspace(w, w * 1.0, n_steps)  # keep w fixed for stability
    # Actually let's slowly explore w:
    w_center = 1.0
    w_var = 0.15
    w_schedule = w_center + w_var * np.cos(np.linspace(0, 4 * np.pi, n_steps)) * np.exp(-np.linspace(0, 3, n_steps))

    momentum = np.zeros(2 * N)
    mass = 0.8
    iu = np.triu_indices(N, k=1)

    start = time.time()
    step_time = time_budget / n_steps

    for step in range(n_steps):
        if time.time() - start > time_budget - 1:
            break
        T = T0 * (T_min / T0) ** (step / n_steps)
        h = 2.0 - w

        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        # Wall forces
        fx -= np.maximum(r - x, 0.0) * 50.0
        fx += np.maximum(x + r - w, 0.0) * 50.0
        fy -= np.maximum(r - y, 0.0) * 50.0
        fy += np.maximum(y + r - h, 0.0) * 50.0
        # Pair forces
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = np.maximum(r[iu[0]] + r[iu[1]] - d, 0.0)
        f_pairs = overlap * 20.0
        ux = dx / d
        uy = dy / d
        for idx in range(len(iu[0])):
            ii, jj = iu[0][idx], iu[1][idx]
            fx[ii] += ux[idx] * f_pairs[idx]
            fy[ii] += uy[idx] * f_pairs[idx]
            fx[jj] -= ux[idx] * f_pairs[idx]
            fy[jj] -= uy[idx] * f_pairs[idx]

        # Random perturbation
        noise = 0.01 * T / T0 + 0.001
        px = noise * rng.standard_normal(N)
        py = noise * rng.standard_normal(N)

        # Momentum update
        momentum[0:N] = mass * momentum[0:N] + (fx + px)
        momentum[N:2*N] = mass * momentum[N:2*N] + (fy + py)

        # Apply
        step_size = 0.005
        x_new = x + step_size * momentum[0:N]
        y_new = y + step_size * momentum[N:2*N]

        # Clip to bounds
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        # Radial growth attempt
        r_new = r.copy()
        grow = 1.0002
        rt = r * grow
        if _is_valid(x_new, y_new, rt, w, h, tol=1e-6):
            r_new = rt
        else:
            # Try smaller growth
            rt2 = r * 1.0001
            if _is_valid(x_new, y_new, rt2, w, h, tol=1e-6):
                r_new = rt2

        # Metropolis acceptance based on energy
        e_old = _compute_energy(x, y, r, w)
        e_new = _compute_energy(x_new, y_new, r_new, w)
        delta_e = e_new - e_old
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            x, y, r = x_new, y_new, r_new
        else:
            # Keep momentum but revert position
            x, y = x_new * 0 + x, y_new * 0 + y  # keep old
            # Actually revert properly
            # (we already have old x,y in variables)

        # Update w slowly
        if step % 10 == 0:
            w_new = w_schedule[step]
            h_new = 2.0 - w_new
            if w_new > 0.3 and h_new > 0.3:
                # Check if still valid with new w
                if _is_valid(x, y, r, w_new, h_new, tol=1e-6):
                    w = w_new

        # Track best
        if _is_valid(x, y, r, w, 2.0 - w, tol=1e-6):
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

    return np.concatenate([best_x, best_y, best_r, [best_w]])


def _force_relax(x, y, r, w, max_iter=200):
    """Force-based relaxation to eliminate overlaps."""
    h = 2.0 - w
    iu = np.triu_indices(N, k=1)
    for it in range(max_iter):
        if h < 0.1 or w < 0.1:
            break
        fx = np.zeros(N)
        fy = np.zeros(N)
        fx -= np.maximum(r - x, 0.0) * 100.0
        fx += np.maximum(x + r - w, 0.0) * 100.0
        fy -= np.maximum(r - y, 0.0) * 100.0
        fy += np.maximum(y + r - h, 0.0) * 100.0
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = np.maximum(r[iu[0]] + r[iu[1]] - d, 0.0)
        max_ov = float(np.max(overlap)) if overlap.size else 0.0
        if max_ov < 1e-9:
            break
        f_pairs = overlap * 15.0
        ux = dx / d
        uy = dy / d
        for idx in range(len(iu[0])):
            ii, jj = iu[0][idx], iu[1][idx]
            fx[ii] += ux[idx] * f_pairs[idx]
            fy[ii] += uy[idx] * f_pairs[idx]
            fx[jj] -= ux[idx] * f_pairs[idx]
            fy[jj] -= uy[idx] * f_pairs[idx]
        dt = 0.003
        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    return x, y, r, w


def _slsqp_polish(z, time_budget):
    """SLSQP polish with penalty objective."""
    n = N

    def objective(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        obj = -np.sum(r)
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = np.maximum(r[iu[0]] + r[iu[1]] - d, 0.0)
        obj += 1000.0 * np.sum(overlap * overlap)
        obj += 1000.0 * (np.sum(np.maximum(r - x, 0.0) ** 2) +
                        np.sum(np.maximum(x + r - w, 0.0) ** 2) +
                        np.sum(np.maximum(r - y, 0.0) ** 2) +
                        np.sum(np.maximum(y + r - h, 0.0) ** 2))
        return obj

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    maxiter = int(min(800, time_budget * 80))
    try:
        res = minimize(objective, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        return res.x
    except Exception:
        return z


def _greedy_expansion(z, time_budget):
    """Per-circle greedy radius expansion via binary search."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    for rnd in range(8):
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
            if r_max > r[i] + 1e-8:
                lo, hi = r[i], r_max
                for _ in range(40):
                    mid = (lo + hi) / 2.0
                    rt = r.copy()
                    rt[i] = mid
                    if _is_valid(x, y, rt, w, h, tol=1e-8):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    if _is_valid(x, y, r, w, h):
        zn = np.empty(3 * N + 1)
        zn[0::3][:n] = x
        zn[1::3][:n] = y
        zn[2::3][:n] = r
        zn[-1] = w
        return zn
    return None


def _contact_graph_optimize(z, time_budget):
    """Optimize using only active (contact) constraints."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    mean_r = np.mean(r)
    gap_thresh = 0.05 * mean_r
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(dx * dx + dy * dy)
    r_sum = r[iu[0]] + r[iu[1]]
    gaps = d - r_sum
    contact_mask = gaps < gap_thresh
    contact_pairs = (iu[0][contact_mask], iu[1][contact_mask])

    if len(contact_pairs[0]) == 0:
        return _greedy_expansion(z, time_budget)

    constraints = []
    for a, b in zip(contact_pairs[0], contact_pairs[1]):
        def c_fn(a=a, b=b):
            return lambda p: ((p[3*a] - p[3*b])**2 + (p[3*a+1] - p[3*b+1])**2
                             - (p[3*a+2] + p[3*b+2])**2)
        constraints.append({'type': 'ineq', 'fun': c_fn()})
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    bounds = [(0, 2)] * n + [(0, 2)] * n + [(0.001, 1.0)] * n + [(0.3, 1.7)]

    def obj(p):
        return -np.sum(p[2::3][:n])

    maxiter = int(min(1500, time_budget * 200))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 200})
        zopt = res.x
        x2 = zopt[0::3][:n]
        y2 = zopt[1::3][:n]
        r2 = zopt[2::3][:n]
        w2 = zopt[-1]
        if _is_valid(x2, y2, r2, w2, 2.0 - w2):
            return zopt
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_value = 0.0
    best_result = None

    def check_validity(z):
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
            if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
                return False, 0.0
            if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                rs = r[i] + r[j]
                if d2 < rs * rs - 1e-9:
                    return False, 0.0
        return True, float(np.sum(r))

    def update_best(z):
        nonlocal best_value, best_result
        if z is None:
            return
        valid, val = check_validity(z)
        if valid and val > best_value:
            best_value = val
            best_result = z.copy()

    # Generate diverse initial configurations
    def get_initial_guesses():
        guesses = []
        side = 5  # ceil(sqrt(21))
        # Regular grid at various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8, 0.9, 1.1]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                z[3 * i] = (col + 0.5) / side * w0
                z[3 * i + 1] = (row + 0.5) / side * h0
                z[3 * i + 2] = min(w0 / side, h0 / side) * 0.45
            z[-1] = w0
            guesses.append(z)
        # Random configurations
        rng = np.random.default_rng(123)
        for _ in range(12):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.13)
                z[3 * i] = rng.uniform(r0, max(r0 + 1e-3, w0 - r0))
                z[3 * i + 1] = rng.uniform(r0, max(r0 + 1e-3, h0 - r0))
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
        # Hexagonal-ish packing
        for w0 in [0.9, 1.0, 1.1]:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            z = np.zeros(3 * N + 1)
            cols = int(np.floor(w0 / 0.28))
            for i in range(N):
                row, col = divmod(i, max(1, cols))
                offset = 0.14 if row % 2 == 1 else 0.0
                z[3 * i] = (col + 0.5 + offset / 0.28) / max(1, cols + 1) * w0
                z[3 * i + 1] = (row + 0.5) / (N / max(1, cols) + 0.5) * h0
                z[3 * i + 2] = 0.13
            z[-1] = w0
            guesses.append(z)
        return guesses

    guesses = get_initial_guesses()
    n_guesses = len(guesses)

    # Phase 1: Annealed momentum for each guess
    anneal_time_each = min(6.0, 80.0 / n_guesses)
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 60:
            break
        t_budget = time_budget = min(anneal_time_each, deadline - time.time() - 60)
        if t_budget < 2.0:
            break
        z_annealed = _anneal_momentum(z0, t_budget, seed=i)
        if z_annealed is not None:
            # Quick SLSQP polish
            if time.time() < deadline - 55:
                z_polished = _slsqp_polish(z_annealed, deadline - time.time() - 55)
                update_best(z_polished)
            update_best(z_annealed)

    # Phase 2: Refine best with contact graph + greedy
    if best_result is not None:
        if time.time() < deadline - 25:
            cg = _contact_graph_optimize(best_result, deadline - time.time() - 25)
            update_best(cg)

        if best_result is not None and time.time() < deadline - 15:
            ge = _greedy_expansion(best_result, deadline - time.time() - 15)
            update_best(ge)

        # More annealing rounds on the best
        if best_result is not None and time.time() < deadline - 30:
            for rnd in range(3):
                if time.time() > deadline - 30:
                    break
                t_b = min(8.0, deadline - time.time() - 30)
                if t_b < 3:
                    break
                z_a = _anneal_momentum(best_result, t_b, seed=1000 + rnd * 7)
                if z_a is not None:
                    if time.time() < deadline - 28:
                        z_p = _slsqp_polish(z_a, deadline - time.time() - 28)
                        update_best(z_p)
                    update_best(z_a)

        # Final SLSQP polish
        if best_result is not None and time.time() < deadline - 12:
            z_final = _slsqp_polish(best_result, deadline - time.time() - 12)
            update_best(z_final)

        # Final greedy
        if best_result is not None and time.time() < deadline - 5:
            ge = _greedy_expansion(best_result, deadline - time.time() - 5)
            update_best(ge)

        # Final contact graph
        if best_result is not None and time.time() < deadline - 3:
            cg = _contact_graph_optimize(best_result, deadline - time.time() - 3)
            update_best(cg)

    if best_result is None:
        # Fallback grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Tiny safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    for _ in range(30):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
