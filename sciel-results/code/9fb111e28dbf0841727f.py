"""Circle packing in perimeter-4 rectangle: Flexible grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _compute_cell_inradii(grid_x, grid_y):
    """Compute inradii for each cell in the grid."""
    NX = len(grid_x) - 1
    NY = len(grid_y) - 1
    inradii = np.zeros((NY, NX))
    for r in range(NY):
        for c in range(NX):
            w_cell = grid_x[c+1] - grid_x[c]
            h_cell = grid_y[r+1] - grid_y[r]
            inradii[r, c] = min(w_cell, h_cell) / 2.0
    return inradii

def _objective_flexible(z, w, h, n_active):
    """
    Objective for flexible grid with variable circle positions.
    z layout: [gx_0..gx_{nx-2}, gy_0..gy_{ny-2}, x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Circles are constrained to stay within their assigned cells.
    """
    nx = int(np.sqrt(n_active))
    ny = n_active // nx
    if ny * nx != n_active:
        # Try different factorization
        for nx_try in range(3, 10):
            if n_active % nx_try == 0:
                nx = nx_try
                ny = n_active // nx
                break
    
    n_gx = nx - 1
    n_gy = ny - 1
    n_vars = n_gx + n_gy + 3 * n_active
    
    if len(z) < n_vars:
        return 1e6
    
    idx = 0
    gx = z[idx:idx+n_gx]; idx += n_gx
    gy = z[idx:idx+n_gy]; idx += n_gy
    cx = z[idx:idx+n_active]; idx += n_active
    cy = z[idx:idx+n_active]; idx += n_active
    radii = z[idx:idx+n_active]
    
    # Grid validation
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    if np.any(gx < 1e-4) or np.any(gx > w - 1e-4):
        return 1e6
    if np.any(gy < 1e-4) or np.any(gy > h - 1e-4):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    inradii = _compute_cell_inradii(grid_x, grid_y)
    
    # Assign circles to cells (row-major order)
    # Circle i goes to cell (i // nx, i % nx)
    cell_r = np.zeros(n_active)
    for i in range(n_active):
        row = i // nx
        col = i % nx
        cell_r[i] = inradii[row, col]
    
    # Constraint: each circle must fit in its cell
    # Circle center must be within [cell_left + r, cell_right - r] x [cell_bottom + r, cell_top - r]
    penalty = 0.0
    for i in range(n_active):
        row = i // nx
        col = i % nx
        x_left = grid_x[col]
        x_right = grid_x[col+1]
        y_bottom = grid_y[row]
        y_top = grid_y[row+1]
        
        r_i = radii[i]
        if r_i < 1e-6:
            continue
        
        # Check if circle fits in cell
        if cx[i] < x_left + r_i - 1e-6 or cx[i] > x_right - r_i + 1e-6:
            penalty += 1000.0 * max(0, x_left + r_i - cx[i]) ** 2
            penalty += 1000.0 * max(0, cx[i] - (x_right - r_i)) ** 2
        if cy[i] < y_bottom + r_i - 1e-6 or cy[i] > y_top - r_i + 1e-6:
            penalty += 1000.0 * max(0, y_bottom + r_i - cy[i]) ** 2
            penalty += 1000.0 * max(0, cy[i] - (y_top - r_i)) ** 2
        
        # Radius cannot exceed inradius
        if r_i > cell_r[i] + 1e-6:
            penalty += 1000.0 * (r_i - cell_r[i]) ** 2
    
    # Overlap penalty between circles
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = radii[:, None] + radii[None, :]
    
    iu = np.triu_indices(n_active, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    penalty += 500.0 * np.sum(overlap * overlap)
    
    total_r = np.sum(radii)
    
    return -total_r + penalty

def _optimize_flexible(w, h, n_active, time_budget):
    """Optimize flexible grid layout."""
    nx = int(np.sqrt(n_active))
    ny = n_active // nx
    if ny * nx != n_active:
        for nx_try in range(3, 10):
            if n_active % nx_try == 0:
                nx = nx_try
                ny = n_active // nx
                break
    
    n_gx = nx - 1
    n_gy = ny - 1
    
    # Initial guess: uniform grid, circles at cell centers
    gx0 = np.linspace(w/nx, w - w/nx, n_gx)
    gy0 = np.linspace(h/ny, h - h/ny, n_gy)
    
    cx0 = np.zeros(n_active)
    cy0 = np.zeros(n_active)
    r0 = np.zeros(n_active)
    
    for i in range(n_active):
        row = i // nx
        col = i % nx
        cx0[i] = (grid_x[col] + grid_x[col+1]) / 2 if False else (w * (col + 0.5) / nx)
        cy0[i] = (h * (row + 0.5) / ny)
        r0[i] = min(w/nx, h/ny) / 2.0 * 0.9
    
    z0 = np.concatenate([gx0, gy0, cx0, cy0, r0])
    
    # Bounds
    bounds = []
    for _ in range(n_gx):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(n_gy):
        bounds.append((1e-4, h - 1e-4))
    for _ in range(n_active):
        bounds.append((1e-4, w - 1e-4))  # cx
    for _ in range(n_active):
        bounds.append((1e-4, h - 1e-4))  # cy
    for _ in range(n_active):
        bounds.append((1e-4, min(w, h)/2.0))  # radii
    
    def obj(z):
        return _objective_flexible(z, w, h, n_active)
    
    maxiter = int(min(500, time_budget * 30))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 150})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    idx = 0
    gx_opt = z_opt[idx:idx+n_gx]; idx += n_gx
    gy_opt = z_opt[idx:idx+n_gy]; idx += n_gy
    cx_opt = z_opt[idx:idx+n_active]; idx += n_active
    cy_opt = z_opt[idx:idx+n_active]; idx += n_active
    r_opt = z_opt[idx:idx+n_active]
    
    return cx_opt, cy_opt, r_opt

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
    
    # Try different aspect ratios and grid configurations
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.3, 0.7, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.35:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        try:
            x, y, r = _optimize_flexible(w, h, N, remaining)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: simple 5x5 grid with 21 circles (drop last 4)
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
    for _ in range(10):
        rt = r * (1 - 1e-7)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
