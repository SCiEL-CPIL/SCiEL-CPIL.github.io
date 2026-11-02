"""Circle packing in perimeter-4 rectangle: force-directed relaxation + L-BFGS-B with log-sum-exp penalty + expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _perimeter(w, h):
    return 2.0 * (w + h)


def _radius_sum(r):
    return float(np.sum(r))


def _is_valid(x, y, r, w, h, tol=1e-7):
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _smooth_push_apart_energy(z, w, h, beta=500.0):
    """
    z = [x0,y0,r0, x1,y1,r1, ..., xN-1,yN-1,rN-1]
    Returns (energy, grad) for a smooth objective.
    """
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()

    energy = 0.0
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    # Wall constraints: push circles inward
    for i in range(n):
        # left wall
        d = x[i] - r[i]
        if d < 0:
            pen = d * d
            energy += pen
            gx[i] += 2 * d
            gr[i] -= 2 * d
        # right wall
        d = w - x[i] - r[i]
        if d < 0:
            pen = d * d
            energy += pen
            gx[i] -= 2 * d
            gr[i] -= 2 * d
        # bottom wall
        d = y[i] - r[i]
        if d < 0:
            pen = d * d
            energy += pen
            gy[i] += 2 * d
            gr[i] -= 2 * d
        # top wall
        d = h - y[i] - r[i]
        if d < 0:
            pen = d * d
            energy += pen
            gy[i] -= 2 * d
            gr[i] -= 2 * d

    # Pairwise overlap: smooth log-sum-exp style
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            d = np.sqrt(max(d2, 1e-12))
            rs = r[i] + r[j]
            gap = d - rs
            if gap < 0:
                # smooth penalty
                t = beta * gap
                if t < -50:
                    log_term = t  # log(exp(t)) ~ t for very negative
                else:
                    log_term = np.log1p(np.exp(t))
                # derivative of log(1+exp(beta*gap)) w.r.t. gap = beta*sigmoid(beta*gap)
                sig = 1.0 / (1.0 + np.exp(-t)) if t > -50 else 0.0
                dEdgap = beta * sig
                # chain rule
                # gap = d - rs, d = sqrt(dx^2+dy^2)
                # d(gap)/d(x[i]) = dx/d = dx/d, d(gap)/d(r[i]) = -1
                if d > 1e-12:
                    gx[i] += dEdgap * dx / d
                    gy[i] += dEdgap * dy / d
                    gx[j] -= dEdgap * dx / d
                    gy[j] -= dEdgap * dy / d
                gr[i] -= dEdgap
                gr[j] -= dEdgap
                energy += log_term
            else:
                # no overlap, but add tiny regularization to encourage separation
                pass

    # Objective: maximize sum of radii -> minimize -sum(r)
    energy -= np.sum(r)
    gr -= 1.0  # gradient of -sum(r) w.r.t. each r is -1

    grad = np.empty(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr

    return energy, grad


def _force_relax(x, y, r, w, h, iterations=200, learning_rate=0.01):
    """Force-directed relaxation to push circles apart and inward."""
    n = N
    for it in range(iterations):
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)

        # Wall forces
        for i in range(n):
            # left
            d = x[i] - r[i]
            if d < 0:
                fx[i] += 10.0 * d
                fr[i] -= 10.0 * d
            # right
            d = w - x[i] - r[i]
            if d < 0:
                fx[i] -= 10.0 * d
                fr[i] -= 10.0 * d
            # bottom
            d = y[i] - r[i]
            if d < 0:
                fy[i] += 10.0 * d
                fr[i] -= 10.0 * d
            # top
            d = h - y[i] - r[i]
            if d < 0:
                fy[i] -= 10.0 * d
                fr[i] -= 10.0 * d

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                d = np.sqrt(max(d2, 1e-12))
                rs = r[i] + r[j]
                gap = d - rs
                if gap < 0:
                    force = 50.0 * (-gap)
                    if d > 1e-12:
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
                    fr[i] -= force
                    fr[j] -= force

        # Update
        for i in range(n):
            x[i] += learning_rate * fx[i]
            y[i] += learning_rate * fy[i]
            r[i] = max(0.01, r[i] + learning_rate * fr[i])
            # Clamp to walls
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

    return x, y, r


def _grid_cell_layout(w, h, seed=0):
    """Create initial layout on a grid."""
    rng = np.random.default_rng(seed)
    n_cols = 5
    n_rows = 5
    cell_w = w / n_cols
    cell_h = h / n_rows
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    idx = 0
    for i in range(n_cols):
        for j in range(n_rows):
            if idx >= N:
                break
            cx = (i + 0.5) * cell_w + rng.uniform(-0.1, 0.1) * cell_w
            cy = (j + 0.5) * cell_h + rng.uniform(-0.1, 0.1) * cell_h
            cx = np.clip(cx, 0, w)
            cy = np.clip(cy, 0, h)
            r_max = min(cx, w - cx, cy, h - cy)
            x[idx] = cx
            y[idx] = cy
            r[idx] = max(0.05, r_max * 0.9)
            idx += 1
    return x, y, r


def _lbfgs_optimize(z0, w, h, time_budget, beta=500.0):
    """Optimize using L-BFGS-B with the smooth objective."""
    n = N

    def objective(zv):
        if time.time() > time_budget:
            return 1e10, np.zeros(3 * n)
        e, g = _smooth_push_apart_energy(zv, w, h, beta)
        return e, g

    max_iter = int(min(3000, (time_budget - time.time()) * 100))
    max_iter = max(100, max_iter)

    bounds = []
    for i in range(n):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
        bounds.append((0.01, min(w, h) / 2))  # r

    try:
        res = minimize(objective, z0, jac=True, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                                'maxls': 50, 'maxcor': 20})
        return res.x
    except Exception:
        return z0


def _slsqp_refine(z0, w, h, time_budget):
    """Refine using SLSQP with explicit constraints."""
    n = N

    def objective(zv):
        x = zv[0::3]
        y = zv[1::3]
        r = zv[2::3]
        if np.any(r <= 0):
            return 1e10
        # Penalize overlaps
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                rs = r[i] + r[j]
                if d2 < rs * rs:
                    pen += (rs - np.sqrt(max(d2, 0))) ** 2
        # Wall violations
        if np.any(x < r) or np.any(x > w - r):
            pen += 100.0 * (np.sum(np.maximum(0, r - x)) + np.sum(np.maximum(0, x + r - w)))
        if np.any(y < r) or np.any(y > h - r):
            pen += 100.0 * (np.sum(np.maximum(0, r - y)) + np.sum(np.maximum(0, y + r - h)))
        return -np.sum(r) + 500.0 * pen

    max_iter = int(min(500, (time_budget - time.time()) * 50))
    max_iter = max(50, max_iter)

    bounds = [(0, w), (0, h), (0.01, min(w, h) / 2)] * n
    try:
        res = minimize(objective, z0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'maxls': 50})
        return res.x
    except Exception:
        return z0


def _expansion_relaxation_polish(x, y, r, w, h, time_budget):
    """Try to expand radii while maintaining validity."""
    n = N
    for iteration in range(100):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            # Try increasing radius of circle i
            for dr in [0.001, 0.0005, 0.0002, 0.0001]:
                r_trial = r.copy()
                r_trial[i] += dr
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    improved = True
                    break
        if not improved:
            break
    return x, y, r


def _high_precision_polish(x, y, r, w, h, time_budget):
    """Final polish with small perturbations and L-BFGS-B."""
    n = N
    best_val = _radius_sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for trial in range(5):
        if time.time() > time_budget:
            break
        # Small random perturbation
        z = np.empty(3 * n)
        z[0::3] = x + np.random.normal(0, 0.001, n)
        z[1::3] = y + np.random.normal(0, 0.001, n)
        z[2::3] = r
        # Clamp
        for i in range(n):
            z[i * 3] = np.clip(z[i * 3], z[i * 3 + 2], w - z[i * 3 + 2])
            z[i * 3 + 1] = np.clip(z[i * 3 + 1], z[i * 3 + 2], h - z[i * 3 + 2])
            z[i * 3 + 2] = max(0.01, z[i * 3 + 2])

        t_budget = time_budget - (time_budget - time.time()) * 0.3
        z_opt = _lbfgs_optimize(z, w, h, t_budget, beta=1000.0)
        x_t = z_opt[0::3]
        y_t = z_opt[1::3]
        r_t = z_opt[2::3]
        if _is_valid(x_t, y_t, r_t, w, h):
            val = _radius_sum(r_t)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x_t.copy(), y_t.copy(), r_t.copy()

    return best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = _radius_sum(r)
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
                return True
        return False

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 0.85, 1.33, 0.75, 1.5, 0.67, 1.7, 0.6, 2.0, 0.55, 1.1, 0.9, 1.4, 0.7, 1.8, 0.5, 2.2, 1.6, 0.8, 1.25, 0.8, 1.15, 0.95]

    for wi, w0 in enumerate(aspect_ratios):
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Generate initial layout
        x, y, r = _grid_cell_layout(w0, h0, seed=wi * 13 + 7)

        # Force relax
        remaining = deadline - (time.time() - start_time)
        if remaining > 20:
            x, y, r = _force_relax(x, y, r, w0, h0, iterations=100, learning_rate=0.005)

        # Validate and update
        if update_best(x, y, r, w0, h0):
            pass

        # L-BFGS-B optimization
        remaining = deadline - (time.time() - start_time)
        if remaining > 25:
            z = np.empty(3 * N)
            z[0::3] = x
            z[1::3] = y
            z[2::3] = r
            t_budget = time.time() + min(remaining - 15, 30)
            z_opt = _lbfgs_optimize(z, w0, h0, t_budget, beta=500.0)
            x_t = z_opt[0::3]
            y_t = z_opt[1::3]
            r_t = z_opt[2::3]
            update_best(x_t, y_t, r_t, w0, h0)

        # SLSQP refinement
        remaining = deadline - (time.time() - start_time)
        if remaining > 20:
            z = np.empty(3 * N)
            z[0::3] = x_t if 'x_t' in dir() else x
            z[1::3] = y_t if 'y_t' in dir() else y
            z[2::3] = r_t if 'r_t' in dir() else r
            t_budget = time.time() + min(remaining - 10, 20)
            z_opt = _slsqp_refine(z, w0, h0, t_budget)
            x_s = z_opt[0::3]
            y_s = z_opt[1::3]
            r_s = z_opt[2::3]
            update_best(x_s, y_s, r_s, w0, h0)

        # Expansion polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            x_e, y_e, r_e = _expansion_relaxation_polish(x_s, y_s, r_s, w0, h0, time.time() + min(remaining - 5, 10))
            update_best(x_e, y_e, r_e, w0, h0)

    # Global polish on best result
    if best_result is not None:
        bx, by, br, bw, bh = best_result
        remaining = deadline - (time.time() - start_time)
        if remaining > 20:
            x_h, y_h, r_h = _high_precision_polish(bx, by, br, bw, bh, time.time() + min(remaining - 5, 20))
            update_best(x_h, y_h, r_h, bw, bh)

        # Final expansion
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            x_f, y_f, r_f = _expansion_relaxation_polish(best_result[0], best_result[1], best_result[2],
                                                         best_result[3], best_result[4],
                                                         time.time() + min(remaining - 3, 8))
            update_best(x_f, y_f, r_f, best_result[3], best_result[4])

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    bx, by, br, bw, bh = best_result

    # Final safety shrink
    for _ in range(10):
        r_trial = br * (1 - 1e-9)
        if _is_valid(bx, by, r_trial, bw, bh):
            br = r_trial
        else:
            break

    return np.column_stack((bx, by, br))
