"""Circle packing in perimeter-4 rectangle: SLSQP with explicit constraints + Polish Pipeline."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def objective(z):
        """Minimize negative sum of radii."""
        r = z[2::3][:N]
        return -np.sum(r)

    def constraints_list():
        """Return list of constraint functions for SLSQP."""
        cons = []
        
        # Non-negativity of radii
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i + 2]})
        
        # Boundary constraints: circles must stay within rectangle
        # Left wall: x_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i + 2]})
        
        # Right wall: x_i + r_i <= w
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[-1] - z[3*i] - z[3*i + 2]})
        
        # Bottom wall: y_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i + 1] - z[3*i + 2]})
        
        # Top wall: y_i + r_i <= h = 2 - w
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 2.0 - z[-1] - z[3*i + 1] - z[3*i + 2]})
        
        # Pairwise non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for i in range(N):
            for j in range(i + 1, N):
                def pair_constraint(z, i=i, j=j):
                    dx = z[3*i] - z[3*j]
                    dy = z[3*i + 1] - z[3*j + 1]
                    r_sum = z[3*i + 2] + z[3*j + 2]
                    return dx * dx + dy * dy - r_sum * r_sum
                cons.append({'type': 'ineq', 'fun': pair_constraint})
        
        # Rectangle width bounds
        cons.append({'type': 'ineq', 'fun': lambda z: z[-1] - 0.3})
        cons.append({'type': 'ineq', 'fun': lambda z: 1.7 - z[-1]})
        
        return cons

    def check_validity(z, tol=1e-9):
        """Check if packing is valid."""
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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.4, 0.9]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.40
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()
    constraints = constraints_list()
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization with explicit constraints
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(800, max(200, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})

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

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 20:
        # First: Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 10)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # Second: Inflation-based radius maximization
        if best_result is not None and time.time() < deadline - 15:
            inflated = _inflation_polish(best_result, deadline - (time.time() - start_time) - 8)
            if inflated is not None:
                valid, val = check_validity(inflated)
                if valid and val > best_value:
                    best_value = val
                    best_result = inflated

        # Third: Local Search with Perturbation
        if best_result is not None and time.time() < deadline - 10:
            local_searched = _local_search_polish(best_result, deadline - (time.time() - start_time) - 5)
            if local_searched is not None:
                valid, val = check_validity(local_searched)
                if valid and val > best_value:
                    best_value = val
                    best_result = local_searched

        # Fourth: Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 5:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # Fifth: Final inflation pass
        if best_result is not None and time.time() < deadline - 2:
            final_inflated = _inflation_polish(best_result, deadline - (time.time() - start_time))
            if final_inflated is not None:
                valid, val = check_validity(final_inflated)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_inflated

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
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with small shrink
    for _ in range(50):
        r_trial = r * (1 - 1e-7)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Vectorized force-directed relaxation in unified parameter space with adaptive step."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_v = np.maximum(r - x, 0.0)
        right_v = np.maximum(x + r - w, 0.0)
        bottom_v = np.maximum(r - y, 0.0)
        top_v = np.maximum(y + r - h, 0.0)
        fx += 100 * (left_v - right_v)
        fy += 100 * (bottom_v - top_v)

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        
        overlap = np.maximum(r_sum - d, 0.0)
        d_safe = np.where(d > 1e-12, d, 1.0)
        force_mag = overlap / d_safe * 15
        
        fx += np.sum(force_mag * dx, axis=1)
        fy += np.sum(force_mag * dy, axis=1)

        # Check convergence
        max_overlap = np.max(overlap)
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_norm = np.max(np.sqrt(fx * fx + fy * fy))
        if force_norm > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_norm < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions to stay within bounds
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with vectorized force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_v = np.maximum(r - x, 0.0)
        right_v = np.maximum(x + r - w, 0.0)
        bottom_v = np.maximum(r - y, 0.0)
        top_v = np.maximum(y + r - h, 0.0)
        fx += 35 * (left_v - right_v)
        fy += 35 * (bottom_v - top_v)

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        
        overlap = np.maximum(r_sum - d, 0.0)
        d_safe = np.where(d > 1e-12, d, 1.0)
        force_mag = overlap / d_safe * 18
        
        fx += np.sum(force_mag * dx, axis=1)
        fy += np.sum(force_mag * dy, axis=1)

        # Adaptive step
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _inflation_polish(z, time_budget):
    """
    Vectorized inflation-based radius maximization using numpy for speed.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Precompute pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)

    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        # Wall constraints
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))

        # Neighbor constraints: r_i <= d_ij - r_j for all j
        r_allowed = d - r[None, :]
        np.fill_diagonal(r_allowed, np.inf)
        r_neighbor = np.min(r_allowed, axis=1)

        r_max = np.minimum(r_wall, r_neighbor)
        r_new = np.maximum(0.001, r_max * 0.9995)

        mask = r_new > r + 1e-10
        if np.any(mask):
            r[mask] = r_new[mask]
            improved = True

        if not improved:
            break

    # Verify validity
    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _local_search_polish(z, time_budget):
    """
    Local search with random perturbations and vectorized inflation to escape local minima.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = np.sum(r)
    rng = np.random.default_rng(123)

    def _vectorized_inflate(x_t, y_t, r_t):
        """Vectorized radius inflation for trial positions."""
        dx = x_t[:, None] - x_t[None, :]
        dy = y_t[:, None] - y_t[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        
        r_wall = np.minimum(np.minimum(x_t, w - x_t), np.minimum(y_t, h - y_t))
        r_allowed = d - r_t[None, :]
        np.fill_diagonal(r_allowed, np.inf)
        r_neighbor = np.min(r_allowed, axis=1)
        r_max = np.minimum(r_wall, r_neighbor)
        return np.maximum(0.001, r_max * 0.999)

    max_iter = int(min(2000, time_budget * 80))
    for _ in range(max_iter):
        if time.time() > time_budget:
            break

        # Random perturbation of 1-2 circles
        n_perturb = rng.integers(1, 3)
        idxs = rng.choice(n, size=n_perturb, replace=False)
        perturb_mag = rng.uniform(0.01, 0.04)
        
        x_trial = x.copy()
        y_trial = y.copy()
        
        for idx in idxs:
            x_trial[idx] += rng.uniform(-perturb_mag, perturb_mag)
            y_trial[idx] += rng.uniform(-perturb_mag, perturb_mag)
        
        # Clamp
        x_trial = np.clip(x_trial, r, w - r)
        y_trial = np.clip(y_trial, r, h - r)

        # Try to inflate radii (vectorized)
        r_trial = _vectorized_inflate(x_trial, y_trial, r.copy())

        if _is_valid(x_trial, y_trial, r_trial, w, h):
            val = np.sum(r_trial)
            if val > best_val:
                best_val = val
                x = x_trial
                y = y_trial
                r = r_trial

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _constraint_refine(z, time_budget):
    """
    High-precision refinement using SLSQP with explicit constraints.
    """
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    def get_constraints():
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda zv, i=i: zv[3*i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda zv, i=i: zv[3*i] - zv[3*i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda zv, i=i: zv[-1] - zv[3*i] - zv[3*i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda zv, i=i: zv[3*i + 1] - zv[3*i + 2]})
            cons.append({'type': 'ineq', 'fun': lambda zv, i=i: 2.0 - zv[-1] - zv[3*i + 1] - zv[3*i + 2]})
        
        for i in range(n):
            for j in range(i + 1, n):
                def pair_constraint(zv, i=i, j=j):
                    dx = zv[3*i] - zv[3*j]
                    dy = zv[3*i + 1] - zv[3*j + 1]
                    r_sum = zv[3*i + 2] + zv[3*j + 2]
                    return dx * dx + dy * dy - r_sum * r_sum
                cons.append({'type': 'ineq', 'fun': pair_constraint})
        
        cons.append({'type': 'ineq', 'fun': lambda zv: zv[-1] - 0.3})
        cons.append({'type': 'ineq', 'fun': lambda zv: 1.7 - zv[-1]})
        return cons

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
    constraints = get_constraints()

    maxiter = int(min(1500, time_budget * 200))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
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
