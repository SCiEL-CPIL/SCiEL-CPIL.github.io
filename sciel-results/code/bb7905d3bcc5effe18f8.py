"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5


def _find_cell(x, y, grid_x, grid_y):
    """Find the cell (col, row) containing point (x, y)."""
    col = 0
    for c in range(len(grid_x) - 2):
        if x >= grid_x[c + 1]:
            col = c + 1
        else:
            break
    if x >= grid_x[-2]:
        col = len(grid_x) - 2
        
    row = 0
    for r in range(len(grid_y) - 2):
        if y >= grid_y[r + 1]:
            row = r + 1
        else:
            break
    if y >= grid_y[-2]:
        row = len(grid_y) - 2
        
    return col, row


def _inradius(cx, cy, x_left, x_right, y_bottom, y_top):
    """Inradius of circle centered at (cx, cy) within the rectangle."""
    return min(cx - x_left, x_right - cx, cy - y_bottom, y_top - cy)


def _compute_radii(x, y, grid_x, grid_y):
    """Compute radii for all circles given positions and grid."""
    r = np.zeros(N)
    for i in range(N):
        col, row = _find_cell(x[i], y[i], grid_x, grid_y)
        x_left = grid_x[col]
        x_right = grid_x[col + 1]
        y_bottom = grid_y[row]
        y_top = grid_y[row + 1]
        r[i] = _inradius(x[i], y[i], x_left, x_right, y_bottom, y_top)
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


def _objective(z, w, h):
    """
    Objective function.
    z layout: [x_0, y_0, ..., x_{N-1}, y_{N-1}, gx_0, ..., gx_{NX-2}, gy_0, ..., gy_{NY-2}]
    """
    x = z[0:N]
    y = z[N:2*N]
    gx = z[2*N:2*N+NX-1]
    gy = z[2*N+NX-1:2*N+NX-1+NY-1]
    
    if w < 0.2 or h < 0.2:
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 0.02) or np.any(np.diff(grid_y) < 0.02):
        return 1e6
    
    r = _compute_radii(x, y, grid_x, grid_y)
    
    if np.any(r <= 0):
        return 1e6
    
    obj = -np.sum(r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += 200.0 * np.sum(overlap * overlap)
    
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += 200.0 * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )
    
    return obj


def _optimize_aspect(w, h, time_budget):
    """Optimize for given aspect ratio."""
    cw = w / NX
    ch = h / NY
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, NX)
        x[i] = (col + 0.5) * cw
        y[i] = (row + 0.5) * ch
    
    gx0 = np.linspace(cw, w - cw, NX - 1)
    gy0 = np.linspace(ch, h - ch, NY - 1)
    
    z0 = np.concatenate([x, y, gx0, gy0])
    
    bounds = []
    for _ in range(N):
        bounds.append((0.01, w - 0.01))
    for _ in range(N):
        bounds.append((0.01, h - 0.01))
    for _ in range(NX - 1):
        bounds.append((0.05, w - 0.05))
    for _ in range(NY - 1):
        bounds.append((0.05, h - 0.05))
    
    def obj(z):
        return _objective(z, w, h)
    
    maxiter = int(min(200, time_budget * 15))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    x_opt = z_opt[0:N]
    y_opt = z_opt[N:2*N]
    gx_opt = z_opt[2*N:2*N+NX-1]
    gy_opt = z_opt[2*N+NX-1:2*N+NX-1+NY-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    r_opt = _compute_radii(x_opt, y_opt, grid_x, grid_y)
    
    valid, val = _check_validity(x_opt, y_opt, r_opt, w, h)
    if valid:
        return x_opt, y_opt, r_opt, val
    
    grid_x_u = np.linspace(0, w, NX + 1)
    grid_y_u = np.linspace(0, h, NY + 1)
    r_u = _compute_radii(x, y, grid_x_u, grid_y_u)
    valid, val = _check_validity(x, y, r_u, w, h)
    if valid:
        return x, y, r_u, val
    
    return None


def _refine(x, y, r, w, h, grid_x, grid_y, time_budget):
    """Refine positions to maximize inradii."""
    x = x.copy()
    y = y.copy()
    
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        for i in range(N):
            col, row = _find_cell(x[i], y[i], grid_x, grid_y)
            x_left = grid_x[col]
            x_right = grid_x[col + 1]
            y_bottom = grid_y[row]
            y_top = grid_y[row + 1]
            
            cx_c = (x_left + x_right) / 2
            cy_c = (y_bottom + y_top) / 2
            r_c = _inradius(cx_c, cy_c, x_left, x_right, y_bottom, y_top)
            
            if r_c > r[i] + 1e-6:
                tx = x.copy()
                ty = y.copy()
                tx[i] = cx_c
                ty[i] = cy_c
                tr = _compute_radii(tx, ty, grid_x, grid_y)
                valid, _ = _check_validity(tx, ty, tr, w, h)
                if valid:
                    x[i] = cx_c
                    y[i] = cy_c
                    r = tr
                    improved = True
                    continue
            
            for dx in [-0.003, 0.003]:
                for dy in [-0.003, 0.003]:
                    nx_i = x[i] + dx
                    ny_i = y[i] + dy
                    if nx_i < x_left + 0.001 or nx_i > x_right - 0.001:
                        continue
                    if ny_i < y_bottom + 0.001 or ny_i > y_top - 0.001:
                        continue
                    
                    tx = x.copy()
                    ty = y.copy()
                    tx[i] = nx_i
                    ty[i] = ny_i
                    tr = _compute_radii(tx, ty, grid_x, grid_y)
                    if tr[i] > r[i] + 1e-6:
                        valid, _ = _check_validity(tx, ty, tr, w, h)
                        if valid:
                            x[i] = nx_i
                            y[i] = ny_i
                            r = tr
                            improved = True
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 5:
            break
        
        result = _optimize_aspect(w, h, remaining)
        if result is not None:
            x, y, r, val = result
            
            gx = np.linspace(w/NX, w - w/NX, NX-1)
            gy = np.linspace(h/NY, h - h/NY, NY-1)
            grid_x = np.concatenate(([0.0], gx, [w]))
            grid_y = np.concatenate(([0.0], gy, [h]))
            
            x, y, r = _refine(x, y, r, w, h, grid_x, grid_y, deadline - 8)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        w, h = 1.0, 1.0
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
        r = np.full(N, 0.09)
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    for _ in range(3):
        rt = r * (1 - 1e-10)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
    
    return np.column_stack((x, y, r))
