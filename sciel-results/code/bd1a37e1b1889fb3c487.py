"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _cell_inradius(cx, cy, w, h, nx, ny):
    """Inradius of the cell containing point (cx, cy) in an nx*ny grid."""
    cw = w / nx
    ch = h / ny
    col = int(np.clip(cx / cw, 0, nx - 1))
    row = int(np.clip(cy / ch, 0, ny - 1))
    x0 = col * cw
    y0 = row * ch
    x1 = x0 + cw
    y1 = y0 + ch
    return min(cx - x0, x1 - cx, cy - y0, y1 - cy)


def _compute_radii(x, y, w, h, nx, ny, scale=1.0):
    """Compute radii as inradius of each circle's cell, scaled."""
    r = np.array([_cell_inradius(x[i], y[i], w, h, nx, ny) for i in range(N)])
    r *= scale
    return r


def _check_validity(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
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
            if d2 < r_sum * r_sum - tol:
                return False, 0.0
    return True, float(np.sum(r))


def _objective(z, penalty_weight=500.0, nx=5, ny=5):
    """Objective function for grid-based optimization."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return 1e6

    r = _compute_radii(x, y, w, h, nx, ny)

    if np.any(r <= 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
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

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax(x, y, r, w, h, nx, ny, max_iter=300):
    """Force-directed relaxation to resolve overlaps while respecting cell constraints."""
    x = x.copy()
    y = y.copy()
    dt = 0.003
    cw = w / nx
    ch = h / ny

    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Recompute radii based on current positions
        r_cur = _compute_radii(x, y, w, h, nx, ny)

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_cur[:, None] + r_cur[None, :]

        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 10.0
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

        # Wall repulsion
        left_viol = np.maximum(r_cur - x, 0.0)
        right_viol = np.maximum(x + r_cur - w, 0.0)
        bottom_viol = np.maximum(r_cur - y, 0.0)
        top_viol = np.maximum(y + r_cur - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Update
        x += dt * fx
        y += dt * fy

        # Clamp to cell boundaries (each circle stays in its cell)
        for i in range(N):
            col = int(np.clip(x[i] / cw, 0, nx - 1))
            row = int(np.clip(y[i] / ch, 0, ny - 1))
            x0 = col * cw
            y0 = row * ch
            x1 = x0 + cw
            y1 = y0 + ch
            # Keep center within safe region of cell
            margin = 0.01
            x[i] = np.clip(x[i], x0 + margin, x1 - margin)
            y[i] = np.clip(y[i], y0 + margin, y1 - margin)

    return x, y


def _expand_radii(x, y, w, h, nx, ny, time_budget):
    """Try to expand radii by moving circles within their cells."""
    x = x.copy()
    y = y.copy()
    r = _compute_radii(x, y, w, h, nx, ny)
    cw = w / nx
    ch = h / ny

    improved = True
    while improved and time.time() < time_budget:
        improved = False
        for i in range(N):
            # Try small perturbations to increase inradius
            col = int(np.clip(x[i] / cw, 0, nx - 1))
            row = int(np.clip(y[i] / ch, 0, ny - 1))
            x0 = col * cw
            y0 = row * ch
            x1 = x0 + cw
            y1 = y0 + ch

            best_r = r[i]
            best_x, best_y = x[i], y[i]

            # Try centering in cell
            cx_center = (x0 + x1) / 2
            cy_center = (y0 + y1) / 2
            r_center = min(cx_center - x0, x1 - cx_center, cy_center - y0, y1 - cy_center)

            if r_center > best_r + 1e-6:
                # Check if moving to center is valid
                trial_x = x.copy()
                trial_y = y.copy()
                trial_x[i] = cx_center
                trial_y[i] = cy_center
                trial_r = _compute_radii(trial_x, trial_y, w, h, nx, ny)
                valid, _ = _check_validity(trial_x, trial_y, trial_r, w, h)
                if valid:
                    x[i] = cx_center
                    y[i] = cy_center
                    r = trial_r
                    improved = True
                    continue

            # Try small steps in 8 directions
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    nx_i = x[i] + dx
                    ny_i = y[i] + dy
                    # Check if still in cell
                    if nx_i < x0 + 0.005 or nx_i > x1 - 0.005:
                        continue
                    if ny_i < y0 + 0.005 or ny_i > y1 - 0.005:
                        continue

                    trial_x = x.copy()
                    trial_y = y.copy()
                    trial_x[i] = nx_i
                    trial_y[i] = ny_i
                    trial_r = _compute_radii(trial_x, trial_y, w, h, nx, ny)
                    if trial_r[i] > best_r + 1e-6:
                        valid, _ = _check_validity(trial_x, trial_y, trial_r, w, h)
                        if valid:
                            x[i] = nx_i
                            y[i] = ny_i
                            best_r = trial_r[i]
                            best_x, best_y = nx_i, ny_i
                            improved = True

        r = _compute_radii(x, y, w, h, nx, ny)

    return x, y, r


def _slsqp_optimize(z0, time_budget, penalty_weight=500.0, nx=5, ny=5):
    """Run SLSQP optimization."""
    def obj(z):
        return _objective(z, penalty_weight, nx, ny)

    bounds = [(0.01, 1.99)] * (2 * N) + [(0.3, 1.7)]
    maxiter = int(min(400, time_budget * 30))

    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        return res.x
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    best_value = 0.0
    best_result = None

    # Define grid configurations (nx, ny) that fit 21 circles
    # 4x6=24, 3x7=21, 6x4=24, 5x5=25, 7x3=21, 2x11=22, 11x2=22, 1x21=21
    grid_configs = [
        (4, 6), (6, 4), (3, 7), (7, 3), (5, 5), (2, 11), (11, 2), (1, 21), (21, 1)
    ]
    
    # Aspect ratios to try for each grid
    # We want to explore w/h ratios that match the grid aspect ratio for near-square cells
    # But also explore deviations
    aspect_ratios = [1.0, 0.8, 1.25, 0.6, 1.67, 0.5, 2.0, 0.4, 2.5, 0.33, 3.0]

    tasks = []
    for nx, ny in grid_configs:
        if nx * ny < N:
            continue
        for ar in aspect_ratios:
            # w/h = ar => w = ar * h, 2w + 2h = 4 => w + h = 2 => h(ar+1) = 2 => h = 2/(ar+1)
            h = 2.0 / (ar + 1.0)
            w = ar * h
            if w < 0.2 or h < 0.2:
                continue
            tasks.append((nx, ny, w))

    # Shuffle tasks to ensure diversity in early time
    rng = np.random.default_rng(123)
    rng.shuffle(tasks)

    for nx, ny, w0 in tasks:
        if time.time() > deadline - 30:
            break

        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        cw = w0 / nx
        ch = h0 / ny
        
        # Initialize positions in first N cells
        for i in range(N):
            row, col = divmod(i, nx)
            cx = (col + 0.5) * cw
            cy = (row + 0.5) * ch
            z[3 * i] = cx
            z[3 * i + 1] = cy
        z[-1] = w0

        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        # Force relaxation
        r_init = _compute_radii(x, y, w, h, nx, ny)
        x_rel, y_rel = _force_relax(x, y, r_init, w, h, nx, ny, max_iter=200)

        # SLSQP optimization
        z_work = np.zeros(3 * N + 1)
        z_work[0::3][:N] = x_rel
        z_work[1::3][:N] = y_rel
        z_work[-1] = w

        remaining = deadline - (time.time() - start_time)
        z_opt = _slsqp_optimize(z_work, remaining, penalty_weight=500.0, nx=nx, ny=ny)

        if z_opt is not None:
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt

            # Force relax again
            r_init = _compute_radii(x_opt, y_opt, w_opt, h_opt, nx, ny)
            x_opt, y_opt = _force_relax(x_opt, y_opt, r_init, w_opt, h_opt, nx, ny, max_iter=200)

            # Expand radii
            x_opt, y_opt, r_opt = _expand_radii(x_opt, y_opt, w_opt, h_opt, nx, ny, deadline - 15)

            valid, val = _check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
            if valid and val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)

    # If no valid result yet, try a simple fallback
    if best_result is None:
        w = 1.0
        h = 1.0
        nx, ny = 5, 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, nx)
            x[i] = (col + 0.5) * w / nx
            y[i] = (row + 0.5) * h / ny
        r = _compute_radii(x, y, w, h, nx, ny)
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w)
        else:
            # Last resort: tiny circles
            w = 1.0
            h = 1.0
            x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
            y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
            r = np.full(N, 0.09)
            valid, val = _check_validity(x, y, r, w, h)
            if valid:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)

    if best_result is None:
        # Absolute fallback
        return np.array([[0.1, 0.1, 0.05]] * N)

    x, y, r, w = best_result
    h = 2.0 - w

    # Final safety: ensure strict validity
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _check_validity(x, y, r_trial, w, h)[0]:
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
