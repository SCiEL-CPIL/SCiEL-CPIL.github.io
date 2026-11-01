"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))

def _cell_bounds(grid_x, grid_y):
    """
    Compute for each cell: left, right, bottom, top.
    Returns arrays of shape (N_CELLS,) for each bound.
    """
    lefts = np.zeros(N_CELLS)
    rights = np.zeros(N_CELLS)
    bottoms = np.zeros(N_CELLS)
    tops = np.zeros(N_CELLS)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            lefts[idx] = grid_x[c]
            rights[idx] = grid_x[c + 1]
            bottoms[idx] = grid_y[r]
            tops[idx] = grid_y[r + 1]
            idx += 1
    return lefts, rights, bottoms, tops

def _objective_full(z, w, h):
    """
    Full objective: optimize grid lines AND circle positions/radii.
    z layout:
      [0 : NX-1]       : internal x grid lines
      [NX-1 : NX-1+NY-1] : internal y grid lines
      [NX-1+NY-1 : NX-1+NY-1+N] : x positions of active circles
      [NX-1+NY-1+N : NX-1+NY-1+2N] : y positions of active circles
      [NX-1+NY-1+2N : NX-1+NY-1+3N] : radii of active circles
    """
    n_gx = NX - 1
    n_gy = NY - 1
    offset_g = n_gx + n_gy
    offset_x = offset_g
    offset_y = offset_g + N
    offset_r = offset_g + 2 * N
    
    gx = z[0:n_gx]
    gy = z[n_gx:offset_g]
    cx = z[offset_x:offset_y]
    cy = z[offset_y:offset_r]
    rad = z[offset_r:offset_r + N]
    
    # Grid validity
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    if np.any(gx < 1e-4) or np.any(gx > w - 1e-4):
        return 1e6
    if np.any(gy < 1e-4) or np.any(gy > h - 1e-4):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
    
    lefts, rights, bottoms, tops = _cell_bounds(grid_x, grid_y)
    
    # Cell constraints: circle must fit in its cell
    # For active cell i (0-indexed among all cells), check:
    # cx[i] - rad[i] >= lefts[i], cx[i] + rad[i] <= rights[i]
    # cy[i] - rad[i] >= bottoms[i], cy[i] + rad[i] <= tops[i]
    cell_pen = 0.0
    for i in range(N):
        cell_idx = ACTIVE_CELLS[i]
        margin_left = lefts[cell_idx] + rad[i] - cx[i]
        margin_right = cx[i] + rad[i] - rights[cell_idx]
        margin_bottom = bottoms[cell_idx] + rad[i] - cy[i]
        margin_top = cy[i] + rad[i] - tops[cell_idx]
        # Penalize violations
        if margin_left > 0:
            cell_pen += 1000.0 * margin_left * margin_left
        if margin_right > 0:
            cell_pen += 1000.0 * margin_right * margin_right
        if margin_bottom > 0:
            cell_pen += 1000.0 * margin_bottom * margin_bottom
        if margin_top > 0:
            cell_pen += 1000.0 * margin_top * margin_top
    
    # Also penalize negative radii
    neg_r = np.maximum(0.0, -rad)
    cell_pen += 1000.0 * np.sum(neg_r * neg_r)
    
    # Overlap penalty between active circles
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = rad[:, None] + rad[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_pen = 1000.0 * np.sum(overlap * overlap)
    
    total_r = np.sum(rad)
    
    return -total_r + cell_pen + overlap_pen

def _optimize_full(w, h, time_budget):
    """Optimize grid lines and circle positions/radii jointly."""
    # Initial guess: uniform grid, circles centered in cells
    gx0 = np.linspace(w / NX, w - w / NX, NX - 1)
    gy0 = np.linspace(h / NY, h - h / NY, NY - 1)
    
    grid_x0 = np.concatenate(([0.0], gx0, [w]))
    grid_y0 = np.concatenate(([0.0], gy0, [h]))
    
    lefts, rights, bottoms, tops = _cell_bounds(grid_x0, grid_y0)
    
    cx0 = np.zeros(N)
    cy0 = np.zeros(N)
    rad0 = np.zeros(N)
    for i in range(N):
        cell_idx = ACTIVE_CELLS[i]
        cx0[i] = (lefts[cell_idx] + rights[cell_idx]) / 2.0
        cy0[i] = (bottoms[cell_idx] + tops[cell_idx]) / 2.0
        rad0[i] = min(rights[cell_idx] - lefts[cell_idx], tops[cell_idx] - bottoms[cell_idx]) / 2.0
    
    z0 = np.concatenate([gx0, gy0, cx0, cy0, rad0])
    
    n_gx = NX - 1
    n_gy = NY - 1
    offset_g = n_gx + n_gy
    offset_x = offset_g
    offset_y = offset_g + N
    offset_r = offset_g + 2 * N
    
    bounds = []
    for _ in range(n_gx):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(n_gy):
        bounds.append((1e-4, h - 1e-4))
    for i in range(N):
        cell_idx = ACTIVE_CELLS[i]
        # x bounds: must be within cell, but cell boundaries depend on grid
        # Use loose bounds; the penalty handles tight constraints
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))
    for i in range(N):
        bounds.append((1e-4, min(w, h) / 2.0))
    
    def obj(z):
        return _objective_full(z, w, h)
    
    maxiter = int(min(500, time_budget * 30))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    gx_opt = z_opt[0:n_gx]
    gy_opt = z_opt[n_gx:offset_g]
    cx_opt = z_opt[offset_x:offset_y]
    cy_opt = z_opt[offset_y:offset_r]
    rad_opt = z_opt[offset_r:offset_r + N]
    
    return cx_opt, cy_opt, rad_opt

def _refine_local(x, y, r, w, h, time_budget):
    """
    Local refinement: given initial positions and radii, optimize them directly
    without grid constraints, just ensuring circles stay in [0,w]x[0,h] and don't overlap.
    """
    def obj(z):
        cx = z[0:N]
        cy = z[N:2*N]
        rad = z[2*N:3*N]
        
        pen = 0.0
        # Boundary constraints
        for i in range(N):
            if cx[i] - rad[i] < 0:
                pen += 1000.0 * (cx[i] - rad[i]) ** 2
            if cx[i] + rad[i] > w:
                pen += 1000.0 * (cx[i] + rad[i] - w) ** 2
            if cy[i] - rad[i] < 0:
                pen += 1000.0 * (cy[i] - rad[i]) ** 2
            if cy[i] + rad[i] > h:
                pen += 1000.0 * (cy[i] + rad[i] - h) ** 2
            if rad[i] < 0:
                pen += 1000.0 * rad[i] ** 2
        
        # Overlap
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rad[:, None] + rad[None, :]
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        pen += 1000.0 * np.sum(overlap * overlap)
        
        return -np.sum(rad) + pen
    
    z0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))
    for i in range(N):
        bounds.append((1e-4, min(w, h) / 2.0))
    
    maxiter = int(min(300, time_budget * 50))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})
        if res.success or res.fun < obj(z0):
            z_opt = res.x
            return z_opt[0:N].copy(), z_opt[N:2*N].copy(), z_opt[2*N:3*N].copy()
    except Exception:
        pass
    
    return x.copy(), y.copy(), r.copy()

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

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 0.95, 1.05, 0.85, 1.18]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        try:
            # Phase 1: Grid-based optimization
            t0 = time.time()
            x, y, r = _optimize_full(w, h, remaining * 0.5)
            t_grid = time.time() - t0
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
            
            # Phase 2: Local refinement without grid constraints
            remaining2 = deadline - (time.time() - start_time)
            if remaining2 > 5:
                x2, y2, r2 = _refine_local(x, y, r, w, h, remaining2 * 0.8)
                valid2, val2 = _check_validity(x2, y2, r2, w, h)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = (x2.copy(), y2.copy(), r2.copy(), w)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to simple grid
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
    
    # Final slight shrink to ensure validity
    for _ in range(5):
        rt = r * (1 - 1e-8)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
