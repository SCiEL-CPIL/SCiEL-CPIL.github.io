"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Combined: cellular init + adaptive penalty SLSQP + force relaxation + expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=5000.0):
    """Vectorized objective with smooth quadratic penalties for overlaps and boundaries."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.25 or h < 0.25 or np.any(r < 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-16))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Aspect ratio soft constraint
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _cellular_init(w0, h0, n_rows, n_cols):
    """Create initial guess using cellular decomposition of the rectangle."""
    z = np.zeros(3 * N + 1)
    z[-1] = w0
    
    # Place circles in a grid pattern within cells
    cell_w = w0 / n_cols
    cell_h = h0 / n_rows
    
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            r0 = min(cell_w, cell_h) * 0.45
            z[3 * count] = cx
            z[3 * count + 1] = cy
            z[3 * count + 2] = r0
            count += 1
    
    # If we have fewer than N, add remaining in remaining cells
    while count < N:
        row = count // n_cols
        col = count % n_cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        r0 = min(cell_w, cell_h) * 0.35
        z[3 * count] = cx
        z[3 * count + 1] = cy
        z[3 * count + 2] = r0
        count += 1
    
    return z


def _force_relax(z, max_iter=800, time_budget=15.0):
    """Force-directed relaxation in unified space (x, y, r, w) with better damping."""
    z = z.copy().astype(float)
    t0 = time.time()
    dt = 0.002
    min_dt = 1e-6
    damping = 0.9

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break

        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        fx -= left_viol * 80.0
        fx += right_viol * 80.0
        fy -= bottom_viol * 80.0
        fy += top_viol * 80.0

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-14))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        max_overlap = np.max(overlap) if len(overlap) > 0 else 0.0
        max_wall = max(np.max(left_viol), np.max(right_viol),
                       np.max(bottom_viol), np.max(top_viol))

        if max_overlap < 1e-9 and max_wall < 1e-9:
            break

        if np.max(overlap) > 1e-8:
            f_pairs = overlap * 40.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]
                fr[i_idx] -= overlap[idx] * 3.0
                fr[j_idx] -= overlap[idx] * 3.0

        # Update with damping
        z[0::3][:N] += dt * damping * fx
        z[1::3][:N] += dt * damping * fy
        z[2::3][:N] += dt * damping * fr

        # Clamp
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        r = np.clip(r, 0.005, min(w, h) * 0.45)
        z[2::3][:N] = r
        z[0::3][:N] = np.clip(x, r, w - r)
        z[1::3][:N] = np.clip(y, r, h - r)

        # Adaptive step with oscillation detection
        force_mag = np.max(np.sqrt(fx * fx + fy * fy))
        if force_mag > 3.0:
            dt = max(min_dt, dt * 0.7)
        elif force_mag < 0.1:
            dt = min(0.015, dt * 1.1)
        else:
            dt = min(0.015, dt * 1.02)

    return z


def _expand_radii(x, y, r, w, h, max_iter=150):
    """Iteratively expand each radius to its maximum allowed value."""
    r = r.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r_max, 0.001)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r


def _is_valid(cx, cy, r, W, H, tol=1e-9):
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


def _sum_radii(cx, cy, r, W, H):
    if _is_valid(cx, cy, r, W, H):
        return float(np.sum(r))
    return -1.0


def _constraint_refine(z, time_budget=10.0):
    """SLSQP with explicit inequality constraints for final polish."""
    t0 = time.time()
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    z_vec = np.concatenate([x, y, r])

    def objective(p):
        return -np.sum(p[2 * n:3 * n])

    bounds = []
    for i in range(n):
        bounds.extend([(0, w), (0, h), (0.001, min(w, h) / 2.0)])

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[3 * n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: w - p[i] - p[3 * n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n + i] - p[3 * n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: h - p[n + i] - p[3 * n + i]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[i] - p[j]) ** 2 + (p[n + i] - p[n + j]) ** 2 - (p[3 * n + i] + p[3 * n + j]) ** 2
            })

    try:
        res = minimize(objective, z_vec, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 100})
        if time.time() - t0 > time_budget:
            return z
        if res.success or res.fun < objective(z_vec):
            x_opt = res.x[:n]
            y_opt = res.x[n:2 * n]
            r_opt = res.x[2 * n:3 * n]
            if _is_valid(x_opt, y_opt, r_opt, w, h):
                z_out = z.copy()
                z_out[0::3][:n] = x_opt
                z_out[1::3][:n] = y_opt
                z_out[2::3][:n] = r_opt
                return z_out
    except Exception:
        pass
    return z


def _expansion_polish(z, time_budget=10.0):
    """Iterative expansion + re-optimization loop to squeeze more radius."""
    t0 = time.time()
    best_z = z.copy()
    best_val = _sum_radii(z[0::3][:N], z[1::3][:N], z[2::3][:N], z[-1], 2.0 - z[-1])

    for iteration in range(8):
        if time.time() - t0 > time_budget:
            break

        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        # Expand radii
        r = _expand_radii(x, y, r, w, h)

        # Try force relaxation to re-optimize positions with new radii
        z_new = z.copy()
        z_new[2::3][:N] = r
        z_new = _force_relax(z_new, max_iter=400, time_budget=2.5)

        val = _sum_radii(z_new[0::3][:N], z_new[1::3][:N], z_new[2::3][:N], z_new[-1], 2.0 - z_new[-1])
        if val > best_val:
            best_val = val
            best_z = z_new.copy()

        z = z_new

    return best_z


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    def get_initial_guesses():
        guesses = []
        
        # Cellular-based inits with different aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.25, 0.8, 1.4, 0.7]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            # Try different grid configurations
            for n_rows, n_cols in [(4, 5), (5, 4), (3, 7), (7, 3), (6, 4), (4, 6)]:
                if n_rows * n_cols >= N:
                    z = _cellular_init(w0, h0, n_rows, n_cols)
                    guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
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
    
    # Adaptive penalty: start lower, increase for later runs
    base_penalty = 3000.0

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        penalty_weight = base_penalty * (1.0 + i * 0.1)

        def objective(z):
            return _vectorized_objective(z, penalty_weight)

        # Phase 1: Force relaxation
        z_relaxed = _force_relax(z0, max_iter=500, time_budget=6.0)

        # Phase 2: SLSQP with penalty objective
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(80, remaining * 25)))
        try:
            res = minimize(objective, z_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})

            z_opt = res.x.copy()

            # Validate
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w

            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            else:
                # Try repair: shrink radii
                z_repair = z_opt.copy()
                z_repair[2::3][:N] *= 0.90
                x = z_repair[0::3][:N]
                y = z_repair[1::3][:N]
                r = z_repair[2::3][:N]
                w = z_repair[-1]
                h = 2.0 - w
                if _is_valid(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 3: Polish best result
    if best_result is not None and time.time() < deadline - 35:
        # Constraint-based refinement
        polished = _constraint_refine(best_result, time_budget=15.0)
        x = polished[0::3][:N]
        y = polished[1::3][:N]
        r = polished[2::3][:N]
        w = polished[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = polished

    if best_result is not None and time.time() < deadline - 20:
        # Expansion polish
        expanded = _expansion_polish(best_result, time_budget=12.0)
        x = expanded[0::3][:N]
        y = expanded[1::3][:N]
        r = expanded[2::3][:N]
        w = expanded[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = expanded

    if best_result is not None and time.time() < deadline - 10:
        # Final constraint refine
        final = _constraint_refine(best_result, time_budget=8.0)
        x = final[0::3][:N]
        y = final[1::3][:N]
        r = final[2::3][:N]
        w = final[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = final

    if best_result is not None and time.time() < deadline - 5:
        # Final expansion
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        r = _expand_radii(x, y, r, w, h)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = best_result.copy()
                best_result[2::3][:N] = r

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        best_result = np.column_stack([x, y, r])

    # Extract output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(30):
        r_trial = r * (1 - 1e-11)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
