"""Circle packing in perimeter-4 rectangle: Cellular decomposition + local optimization."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

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


def _cell_inradius(x, y, w, h, cell_w, cell_h):
    """Inradius of a rectangular cell at position (x, y) with dimensions (w, h)."""
    return min(x, w - x, y, h - y)


def _generate_grid_layouts(w, h, time_budget):
    """Generate various grid-based layouts for the rectangle."""
    layouts = []
    
    # Try different grid configurations
    for rows in range(3, 8):
        if time.time() > time_budget:
            break
        for cols in range(3, 8):
            if rows * cols < N:
                continue
            if rows * cols > N + 5:  # Don't use too many cells
                continue
            
            cell_w = w / cols
            cell_h = h / rows
            
            # Generate positions for N circles
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            # Fill cells row by row
            idx = 0
            for row in range(rows):
                for col in range(cols):
                    if idx >= N:
                        break
                    x[idx] = (col + 0.5) * cell_w
                    y[idx] = (row + 0.5) * cell_h
                    r[idx] = min(cell_w, cell_h) * 0.45
                    idx += 1
            
            if idx == N:
                # Check validity
                if _is_valid(x, y, r, w, h):
                    layouts.append((x.copy(), y.copy(), r.copy()))
    
    return layouts


def _force_relax(x, y, r, w, h, max_iter=1000, dt_init=0.002):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 15.0
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _polish_and_grow(x, y, r, w, h, time_budget):
    """Polish positions and grow radii."""
    n = N
    x = x.copy()
    y = y.copy()

    # Force-directed polish
    max_iter = int(min(3000, time_budget * 150))
    dt = 0.001
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
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

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization of positions and radii."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    def objective(zv):
        xv = zv[0::3][:n]
        yv = zv[1::3][:n]
        rv = zv[2::3][:n]
        wv = zv[-1]
        hv = 2.0 - wv

        if wv < 0.2 or hv < 0.2 or np.any(rv <= 0):
            return 1e8

        obj = -np.sum(rv)

        # Pairwise overlap penalty
        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        obj += 2000.0 * np.sum(overlap * overlap)

        # Boundary penalty
        left_viol = np.maximum(rv - xv, 0.0)
        right_viol = np.maximum(xv + rv - wv, 0.0)
        bottom_viol = np.maximum(rv - yv, 0.0)
        top_viol = np.maximum(yv + rv - hv, 0.0)
        obj += 2000.0 * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                         np.sum(bottom_viol**2) + np.sum(top_viol**2))

        return obj

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.3, 1.7)]
    maxiter = int(min(800, time_budget * 60))
    
    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            # Try to repair
            r_repair = r_opt * 0.98
            if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try various aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.65, 1.54, 0.58, 1.72, 0.53, 0.92, 1.09]

    for w0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Generate grid layouts
        layouts = _generate_grid_layouts(w0, h0, deadline - 5)

        for x_init, y_init, r_init in layouts:
            if time.time() > deadline - 20:
                break

            # Force relaxation
            x_rel, y_rel = _force_relax(x_init, y_init, r_init, w0, h0, max_iter=600)

            if not _is_valid(x_rel, y_rel, r_init, w0, h0):
                r_shrink = r_init * 0.92
                if not _is_valid(x_rel, y_rel, r_shrink, w0, h0):
                    continue
                r_init = r_shrink

            # Polish and grow
            polished = _polish_and_grow(x_rel, y_rel, r_init, w0, h0, deadline - 15)
            if polished is not None:
                x_p, y_p, r_p = polished
                val = float(np.sum(r_p))
                if val > best_value:
                    best_value = val
                    best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w0, h0)

            # SLSQP optimization if time permits
            if time.time() < deadline - 30 and best_result is not None:
                x_b, y_b, r_b, w_b, h_b = best_result
                opt_result = _slsqp_optimize(x_b, y_b, r_b, w_b, h_b, deadline - 25)
                if opt_result is not None:
                    x_o, y_o, r_o, w_o, h_o = opt_result
                    val = float(np.sum(r_o))
                    if val > best_value:
                        best_value = val
                        best_result = (x_o.copy(), y_o.copy(), r_o.copy(), w_o, h_o)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity
    for _ in range(20):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
