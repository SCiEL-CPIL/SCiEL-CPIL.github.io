"""Circle packing in perimeter-4 rectangle: Improved Grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _solve_grid(w, h, nx, ny):
    """
    Optimize grid lines for a rectangle w x h with nx x ny cells.
    Returns best result tuple or None.
    """
    total_cells = nx * ny
    if total_cells < N:
        return None, 0.0
        
    n_vars = (nx - 1) + (ny - 1)
    
    # Initial guess: uniform
    gx0 = np.linspace(w / nx, w - w / nx, nx - 1)
    gy0 = np.linspace(h / ny, h - h / ny, ny - 1)
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(nx - 1):
        bounds.append((1e-6, w - 1e-6))
    for _ in range(ny - 1):
        bounds.append((1e-6, h - 1e-6))
        
    def objective(z):
        gx = z[0:nx-1]
        gy = z[nx-1:nx-1+ny-1]
        
        if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
            return 1e6
            
        grid_x = np.concatenate(([0.0], gx, [w]))
        grid_y = np.concatenate(([0.0], gy, [h]))
        
        radii = np.zeros(total_cells)
        centers = np.zeros((total_cells, 2))
        
        idx = 0
        for r in range(ny):
            for c in range(nx):
                w_cell = grid_x[c+1] - grid_x[c]
                h_cell = grid_y[r+1] - grid_y[r]
                r_cell = 0.5 * min(w_cell, h_cell)
                cx = (grid_x[c] + grid_x[c+1]) / 2.0
                cy = (grid_y[r] + grid_y[r+1]) / 2.0
                radii[idx] = r_cell
                centers[idx] = [cx, cy]
                idx += 1
                
        # Penalize smallest cells to effectively select top N
        k_inactive = total_cells - N
        if k_inactive > 0:
            sorted_radii = np.sort(radii)
            inactive_sum = np.sum(sorted_radii[:k_inactive])
            penalty = 10.0 * inactive_sum
        else:
            penalty = 0.0
            
        return -np.sum(radii) + penalty

    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 150, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x if (res.success or res.fun < objective(z0)) else z0
    except Exception:
        z_opt = z0
        
    gx_opt = z_opt[0:nx-1]
    gy_opt = z_opt[nx-1:nx-1+ny-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    radii = np.zeros(total_cells)
    centers = np.zeros((total_cells, 2))
    
    idx = 0
    for r in range(ny):
        for c in range(nx):
            w_cell = grid_x[c+1] - grid_x[c]
            h_cell = grid_y[r+1] - grid_y[r]
            r_cell = 0.5 * min(w_cell, h_cell)
            cx = (grid_x[c] + grid_x[c+1]) / 2.0
            cy = (grid_y[r] + grid_y[r+1]) / 2.0
            radii[idx] = r_cell
            centers[idx] = [cx, cy]
            idx += 1
            
    if total_cells > N:
        top_idx = np.argsort(radii)[::-1][:N]
        x = centers[top_idx, 0]
        y = centers[top_idx, 1]
        r = radii[top_idx]
    else:
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
        
    # Validate
    if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
        return None, 0.0
        
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rs = r[i] + r[j]
            if d2 < rs*rs - 1e-9:
                return None, 0.0
                
    val = np.sum(r)
    return (x, y, r, w), val

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_result = None
    best_val = 0.0
    
    grid_shapes = [(3, 7), (7, 3), (4, 6), (6, 4), (5, 5)]
    
    aspect_ratios = np.arange(0.6, 1.41, 0.05)
    
    for nx, ny in grid_shapes:
        if time.time() > deadline - 10:
            break
        for w in aspect_ratios:
            if time.time() > deadline - 5:
                break
            h = 2.0 - w
            if h < 0.3:
                continue
                
            try:
                res, val = _solve_grid(w, h, nx, ny)
                if res is not None and val > best_val:
                    best_val = val
                    best_result = res
            except Exception:
                pass

    if best_result is None:
        w, h = 1.0, 1.0
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
        r = np.full(N, 0.09)
        best_result = (x, y, r, w)
        
    x, y, r, w = best_result
    h = 2.0 - w
    
    for _ in range(3):
        rt = r * (1 - 1e-8)
        valid = True
        if np.any(x < rt - 1e-9) or np.any(x > w - rt + 1e-9) or np.any(y < rt - 1e-9) or np.any(y > h - rt + 1e-9):
             valid = False
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx*dx + dy*dy
                    rs = rt[i] + rt[j]
                    if d2 < rs*rs - 1e-9:
                        valid = False
                        break
                if not valid: break
        if valid:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
