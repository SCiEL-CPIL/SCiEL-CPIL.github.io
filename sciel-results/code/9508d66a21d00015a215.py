"""Circle packing in perimeter-4 rectangle: Cellular decomposition + local optimization."""
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
    """Polish positions and grow radii using binary search for max scale."""
    n = N
    x = x.copy()
    y = y.copy()

    # Force-directed polish
    max_iter = int(min(2000, (time_budget - time.time()) * 200))
    if max_iter < 100:
        max_iter = 100
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

    # Binary search for max uniform scale factor
    lo, hi = 1.0, 1.5
    for _ in range(40):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2
        if _is_valid(x, y, r * mid, w, h):
            lo = mid
        else:
            hi = mid

    r = r * lo * (1 - 1e-10)

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization of positions and radii with smooth penalty."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    def objective(zv):
        xv = zv[0::3][:n]
        yv = zv[1::3][:n]
        rv = np.maximum(zv[2::3][:n], 0.01)
        wv = zv[-1]
        hv = 2.0 - wv

        if wv < 0.25 or hv < 0.25:
            return 1e8

        obj = -np.sum(rv)

        # Smooth pairwise overlap penalty using soft-plus
        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = r_sum[iu] - d_pairs
        # Soft-plus: smooth approximation of max(0, x)
        soft_overlap = np.logaddexp(0, overlap / 0.01) * 0.01
        obj += 3000.0 * np.sum(soft_overlap ** 2)

        # Smooth boundary penalty
        left_viol = rv - xv
        right_viol = xv + rv - wv
        bottom_viol = rv - yv
        top_viol = yv + rv - hv
        soft_l = np.logaddexp(0, left_viol / 0.01) * 0.01
        soft_r = np.logaddexp(0, right_viol / 0.01) * 0.01
        soft_b = np.logaddexp(0, bottom_viol / 0.01) * 0.01
        soft_t = np.logaddexp(0, top_viol / 0.01) * 0.01
        obj += 3000.0 * (np.sum(soft_l**2) + np.sum(soft_r**2) +
                         np.sum(soft_b**2) + np.sum(soft_t**2))

        return obj

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.3, 1.7)]
    remaining = time_budget - time.time()
    maxiter = int(min(500, max(50, remaining * 40)))
    
    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 50})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = np.maximum(z_opt[2::3][:n], 0.01)
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        # Validate and repair
        for scale in [1.0, 0.99, 0.98, 0.95]:
            r_test = r_opt * scale
            if _is_valid(x_opt, y_opt, r_test, w_opt, h_opt):
                return x_opt, y_opt, r_test, w_opt, h_opt
    except Exception:
        pass

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try various aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.65, 1.54, 0.58, 0.92, 1.09]

    for w0 in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Generate grid layouts
        layouts = _generate_grid_layouts(w0, h0, deadline - 10)

        for x_init, y_init, r_init in layouts:
            if time.time() > deadline - 12:
                break

            # Force relaxation
            x_rel, y_rel = _force_relax(x_init, y_init, r_init, w0, h0, max_iter=500)

            if not _is_valid(x_rel, y_rel, r_init, w0, h0):
                r_shrink = r_init * 0.92
                if not _is_valid(x_rel, y_rel, r_shrink, w0, h0):
                    continue
                r_init = r_shrink

            # Polish and grow with binary search
            polished = _polish_and_grow(x_rel, y_rel, r_init, w0, h0, deadline - 10)
            if polished is not None:
                x_p, y_p, r_p = polished
                val = float(np.sum(r_p))
                if val > best_value:
                    best_value = val
                    best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w0, h0)

            # SLSQP optimization per layout if time permits
            if time.time() < deadline - 20:
                opt_result = _slsqp_optimize(x_p, y_p, r_p, w0, h0, deadline - 8)
                if opt_result is not None:
                    x_o, y_o, r_o, w_o, h_o = opt_result
                    val = float(np.sum(r_o))
                    if val > best_value:
                        best_value = val
                        best_result = (x_o.copy(), y_o.copy(), r_o.copy(), w_o, h_o)

    # Final SLSQP polish on best result if time permits
    if best_result is not None and time.time() < deadline - 10:
        x_b, y_b, r_b, w_b, h_b = best_result
        opt_result = _slsqp_optimize(x_b, y_b, r_b, w_b, h_b, deadline - 5)
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
