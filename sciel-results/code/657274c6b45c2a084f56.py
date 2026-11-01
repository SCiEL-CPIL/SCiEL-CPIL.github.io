"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with free circle positions."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))

def _compute_cell_bounds(grid_x, grid_y):
    """Compute cell boundaries and inradii for all cells."""
    n_cells = N_CELLS
    cell_xmin = np.zeros(n_cells)
    cell_xmax = np.zeros(n_cells)
    cell_ymin = np.zeros(n_cells)
    cell_ymax = np.zeros(n_cells)
    inradii = np.zeros(n_cells)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            x_left = grid_x[c]
            x_right = grid_x[c + 1]
            y_bottom = grid_y[r]
            y_top = grid_y[r + 1]
            
            cell_xmin[idx] = x_left
            cell_xmax[idx] = x_right
            cell_ymin[idx] = y_bottom
            cell_ymax[idx] = y_top
            
            w_cell = x_right - x_left
            h_cell = y_top - y_bottom
            inradii[idx] = min(w_cell, h_cell) / 2.0
            
            idx += 1
            
    return cell_xmin, cell_xmax, cell_ymin, cell_ymax, inradii

def _objective_full(z, w, h):
    """
    Full objective: optimize grid lines AND circle positions within cells.
    z layout: [gx_0..gx_{NX-2}, gy_0..gy_{NY-2}, px_0..px_{N-1}, py_0..py_{N-1}]
    """
    n_gx = NX - 1
    n_gy = NY - 1
    
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    
    # Circle positions: px[i], py[i] for i in 0..N-1
    offset = n_gx + n_gy
    px = z[offset:offset + N]
    py = z[offset + N:offset + 2*N]
    
    # Check grid validity
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
    
    cell_xmin, cell_xmax, cell_ymin, cell_ymax, inradii = _compute_cell_bounds(grid_x, grid_y)
    
    # For each active cell, the circle must fit within the cell
    # Circle i is in cell i (0-indexed)
    # r_i = min(px[i] - cell_xmin[i], cell_xmax[i] - px[i], py[i] - cell_ymin[i], cell_ymax[i] - py[i])
    
    # Compute radii based on position within cell
    radii = np.zeros(N)
    for i in range(N):
        r_left = px[i] - cell_xmin[i]
        r_right = cell_xmax[i] - px[i]
        r_bottom = py[i] - cell_ymin[i]
        r_top = cell_ymax[i] - py[i]
        radii[i] = min(r_left, r_right, r_bottom, r_top)
    
    total_r = np.sum(radii)
    
    # Penalty if any radius is negative (circle out of cell)
    neg_penalty = 1000.0 * np.sum(np.maximum(-radii, 0.0) ** 2)
    
    # Overlap penalty between circles
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = radii[:, None] + radii[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    overlap_penalty = 1000.0 * np.sum(overlap * overlap)
    
    # Penalize inactive cells to encourage shrinking
    inactive_idx = np.array(INACTIVE_CELLS)
    inactive_radii = inradii[inactive_idx]
    inactive_penalty = 5.0 * np.sum(inactive_radii)
    
    return -total_r + neg_penalty + overlap_penalty + inactive_penalty

def _optimize_packing(w, h, time_budget, seed=0):
    """Optimize grid and circle positions for given aspect ratio."""
    rng = np.random.RandomState(seed)
    
    # Initial guess
    gx0 = np.linspace(w/NX, w - w/NX, NX - 1)
    gy0 = np.linspace(h/NY, h - h/NY, NY - 1)
    
    # Initial circle positions: center of each cell
    grid_x0 = np.concatenate(([0.0], gx0, [w]))
    grid_y0 = np.concatenate(([0.0], gy0, [h]))
    
    px0 = np.zeros(N)
    py0 = np.zeros(N)
    for i in range(N):
        row = i // NX
        col = i % NX
        px0[i] = (grid_x0[col] + grid_x0[col + 1]) / 2.0
        py0[i] = (grid_y0[row] + grid_y0[row + 1]) / 2.0
    
    z0 = np.concatenate([gx0, gy0, px0, py0])
    
    # Bounds
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
    for _ in range(N):
        bounds.append((1e-6, w - 1e-6))
    for _ in range(N):
        bounds.append((1e-6, h - 1e-6))
    
    def obj(z):
        return _objective_full(z, w, h)
    
    maxiter = int(min(200, max(50, time_budget * 15)))
    
    try:
        res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    n_gx = NX - 1
    n_gy = NY - 1
    gx_opt = z_opt[0:n_gx]
    gy_opt = z_opt[n_gx:n_gx + n_gy]
    px_opt = z_opt[n_gx + n_gy:n_gx + n_gy + N]
    py_opt = z_opt[n_gx + n_gy + N:n_gx + n_gy + 2*N]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    cell_xmin, cell_xmax, cell_ymin, cell_ymax, inradii = _compute_cell_bounds(grid_x, grid_y)
    
    radii = np.zeros(N)
    for i in range(N):
        r_left = px_opt[i] - cell_xmin[i]
        r_right = cell_xmax[i] - px_opt[i]
        r_bottom = py_opt[i] - cell_ymin[i]
        r_top = cell_ymax[i] - py_opt[i]
        radii[i] = min(r_left, r_right, r_bottom, r_top)
    
    return px_opt, py_opt, radii, grid_x, grid_y

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
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 1.2, 0.83]
    
    for trial, w in enumerate(aspect_ratios):
        if time.time() > deadline - 8:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        # Try multiple seeds for diversity
        n_seeds = 3
        per_seed_time = remaining / n_seeds
        
        for seed in range(n_seeds):
            if time.time() > deadline - 5:
                break
            seed_time = deadline - (time.time() - start_time) - 5
            if seed_time < 1:
                break
            
            try:
                x, y, r, grid_x, grid_y = _optimize_packing(w, h, seed_time, seed=seed + trial * 10)
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
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
