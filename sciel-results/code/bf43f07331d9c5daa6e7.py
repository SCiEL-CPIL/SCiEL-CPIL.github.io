"""Circle packing in perimeter-4 rectangle: multi-start SLSQP with constraint-based formulation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


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


def _sum_radii(x, y, r):
    return float(np.sum(r))


def _make_constraints(w, h):
    """Create constraint functions for SLSQP."""
    n = N
    cons = []

    # Wall constraints: x_i - r_i >= 0, etc.
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2], 'jac': None})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2], 'jac': None})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i, w=w: w - z[3*i] - z[3*i+2], 'jac': None})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i, h=h: h - z[3*i+1] - z[3*i+2], 'jac': None})

    # Pairwise non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    iu = np.triu_indices(n, k=1)
    for idx in range(len(iu[0])):
        i_idx = iu[0][idx]
        j_idx = iu[1][idx]
        cons.append({
            'type': 'ineq',
            'fun': lambda z, i=i_idx, j=j_idx: 
                (z[3*i] - z[3*j])**2 + (z[3*i+1] - z[3*j+1])**2 - (z[3*i+2] + z[3*j+2])**2,
            'jac': None
        })

    return cons


def _slsqp_solve(x0, y0, r0, w, h, time_budget):
    """Solve SLSQP optimization for fixed rectangle dimensions."""
    n = N
    z0 = np.zeros(3 * n)
    z0[0::3] = x0
    z0[1::3] = y0
    z0[2::3] = r0

    bounds = [(0.001, w - 0.001)] * n + [(0.001, h - 0.001)] * n + [(0.001, min(w, h)/2)] * n

    def objective(z):
        return -np.sum(z[2::3])

    constraints = _make_constraints(w, h)

    remaining = time_budget - time.time()
    if remaining < 2:
        return None

    maxiter = int(min(500, max(50, remaining * 25)))

    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 50})
        z_opt = res.x
        x_opt = z_opt[0::3]
        y_opt = z_opt[1::3]
        r_opt = z_opt[2::3]

        if _is_valid(x_opt, y_opt, r_opt, w, h):
            return x_opt, y_opt, r_opt
        else:
            # Try slight radius reduction
            for factor in [0.999, 0.998, 0.995, 0.99]:
                r_repair = r_opt * factor
                if _is_valid(x_opt, y_opt, r_repair, w, h):
                    return x_opt, y_opt, r_repair
    except Exception:
        pass
    return None


def _jiggle_polish(x, y, r, w, h, time_budget, n_iter=500):
    """Jiggle positions slightly and try to grow radii."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    best_sum = _sum_radii(x, y, r)

    for it in range(n_iter):
        if time.time() > time_budget:
            break
        # Small random perturbation
        dx = np.random.normal(0, 0.002, n)
        dy = np.random.normal(0, 0.002, n)
        x_new = np.clip(x + dx, r, w - r)
        y_new = np.clip(y + dy, r, h - r)

        if _is_valid(x_new, y_new, r, w, h):
            x, y = x_new, y_new

        # Try to grow all radii slightly
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            new_sum = _sum_radii(x, y, r)
            if new_sum > best_sum:
                best_sum = new_sum

    return x, y, r


def _grow_radii(x, y, r, w, h, time_budget):
    """Iteratively grow radii as much as possible."""
    n = N
    r = r.copy()
    improved = True
    while improved:
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Find max radius for circle i
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_max = min(r_max, d - r[j])
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-6:
                r[i] = r_max
                improved = True
    return r


def _grid_init(w, h, rows, cols, jitter=0.0):
    """Create grid initialization."""
    n = N
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)

    for i in range(n):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * (w / cols)
        y[i] = (row + 0.5) * (h / rows)
        if jitter > 0:
            x[i] += np.random.uniform(-jitter, jitter) * (w / cols)
            y[i] += np.random.uniform(-jitter, jitter) * (h / rows)
        x[i] = np.clip(x[i], 0.02, w - 0.02)
        y[i] = np.clip(y[i], 0.02, h - 0.02)
        r[i] = 0.03

    return x, y, r


def _force_resolve(x, y, r, w, h, max_iter=300):
    """Force-directed method to resolve overlaps."""
    n = N
    dt = 0.002
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        fx -= np.maximum(r - x, 0) * 50
        fx += np.maximum(x + r - w, 0) * 50
        fy -= np.maximum(r - y, 0) * 50
        fy += np.maximum(y + r - h, 0) * 50

        # Pairwise repulsion
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-12))
        overlap = np.maximum(r[iu[0]] + r[iu[1]] - d, 0)

        if np.max(overlap) < 1e-10:
            break

        f = overlap * 10
        ux = dx / d
        uy = dy / d
        for k in range(len(iu[0])):
            i_idx = iu[0][k]
            j_idx = iu[1][k]
            fx[i_idx] += ux[k] * f[k]
            fy[i_idx] += uy[k] * f[k]
            fx[j_idx] -= ux[k] * f[k]
            fy[j_idx] -= uy[k] * f[k]

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    best_value = 0.0
    best_result = None

    # Try multiple aspect ratios and initializations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.5, 0.6, 1.6, 0.55, 1.7, 0.5]

    for w in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h = 2.0 - w
        if h < 0.3:
            continue

        # Try different grid configurations
        grid_configs = [(4, 6), (3, 7), (5, 5), (3, 6), (4, 5), (6, 4), (3, 5), (5, 4)]

        for rows, cols in grid_configs:
            if time.time() > deadline - 10:
                break
            if rows * cols < N:
                continue

            x, y, r = _grid_init(w, h, rows, cols, jitter=0.05)
            x, y = _force_resolve(x, y, r, w, h, max_iter=200)

            if not _is_valid(x, y, r, w, h):
                r *= 0.95
                if not _is_valid(x, y, r, w, h):
                    continue

            # SLSQP optimization
            result = _slsqp_solve(x, y, r, w, h, deadline)
            if result is not None:
                x_opt, y_opt, r_opt = result
                val = _sum_radii(x_opt, y_opt, r_opt)
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w, h)

    # If we have a result, try to polish it
    if best_result is not None and time.time() < deadline - 5:
        x_b, y_b, r_b, w_b, h_b = best_result

        # Jiggle polish
        x_p, y_p, r_p = _jiggle_polish(x_b, y_b, r_b, w_b, h_b, deadline, n_iter=200)
        if _is_valid(x_p, y_p, r_p, w_b, h_b):
            val = _sum_radii(x_p, y_p, r_p)
            if val > best_value:
                best_value = val
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_b, h_b)

        # Grow radii
        if best_result is not None and time.time() < deadline - 2:
            x_b, y_b, r_b, w_b, h_b = best_result
            r_grown = _grow_radii(x_b, y_b, r_b, w_b, h_b, deadline)
            if _is_valid(x_b, y_b, r_grown, w_b, h_b):
                val = _sum_radii(x_b, y_b, r_grown)
                if val > best_value:
                    best_value = val
                    best_result = (x_b.copy(), y_b.copy(), r_grown.copy(), w_b, h_b)

    # Final safety: slightly reduce radii to ensure validity
    if best_result is not None:
        x, y, r, w, h = best_result
        for factor in [1.0, 0.9999, 0.999, 0.998, 0.995]:
            r_check = r * factor
            if _is_valid(x, y, r_check, w, h):
                r = r_check
                break

        return np.column_stack((x, y, r))

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
