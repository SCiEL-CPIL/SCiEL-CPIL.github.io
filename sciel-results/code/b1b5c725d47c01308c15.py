"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _maximize_grid_sum(z, NX, NY, w, h):
    """
    Calculate the negative sum of radii for a given grid configuration.
    z contains the internal grid lines: [gx_0, ..., gx_{NX-2}, gy_0, ..., gy_{NY-2}]
    """
    gx = z[0:NX-1]
    gy = z[NX-1:NX-1+NY-1]
    
    # Check for valid grid (sorted, positive widths/heights)
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    dx = np.diff(grid_x)
    dy = np.diff(grid_y)
    
    if np.any(dx < 1e-6) or np.any(dy < 1e-6):
        return 1e6
        
    # Sum of inradii
    # For each cell (i, j), inradius is 0.5 * min(dx[i], dy[j])
    # Total sum = 0.5 * sum_{i,j} min(dx[i], dy[j])
    # We can compute this efficiently
    # min(dx[i], dy[j]) is a matrix of size (NX, NY)
    min_matrix = np.minimum(dx[:, None], dy[None, :])
    total_r = 0.5 * np.sum(min_matrix)
    
    return -total_r

def _optimize_grid(w, h, NX, NY, time_budget):
    """
    Optimize grid lines for a given rectangle w x h with NX x NY cells.
    """
    # Initial guess: uniform grid
    gx0 = np.linspace(w/NX, w - w/NX, NX - 1)
    gy0 = np.linspace(h/NY, h - h/NY, NY - 1)
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
        
    def obj(z):
        return _maximize_grid_sum(z, NX, NY, w, h)
        
    # Try a few random starts to escape local minima
    best_val = obj(z0)
    best_z = z0.copy()
    
    # Number of random restarts based on time budget
    n_restarts = max(1, int(time_budget / 0.5))
    
    for k in range(n_restarts):
        if time.time() > time_budget:
            break
            
        # Perturb initial guess
        if k == 0:
            z_init = z0.copy()
        else:
            z_init = z0 + np.random.normal(0, 0.1 * max(w, h), size=z0.shape)
            # Clip to bounds
            for i in range(len(z_init)):
                z_init[i] = np.clip(z_init[i], bounds[i][0], bounds[i][1])
                
        try:
            res = minimize(obj, z_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x
        except Exception:
            pass
            
    # Final refinement with SLSQP if needed? Nelder-Mead is often better for non-smooth.
    # Let's try SLSQP on the best result found
    try:
        res2 = minimize(obj, best_z, method='SLSQP', bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12})
        if res2.fun < best_val:
            best_val = res2.fun
            best_z = res2.x
    except Exception:
        pass
        
    gx_opt = best_z[0:NX-1]
    gy_opt = best_z[NX-1:NX-1+NY-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    dx = np.diff(grid_x)
    dy = np.diff(grid_y)
    
    centers = np.zeros((NX * NY, 2))
    radii = np.zeros(NX * NY)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            cx = (grid_x[c] + grid_x[c+1]) / 2.0
            cy = (grid_y[r] + grid_y[r+1]) / 2.0
            r_cell = min(dx[c], dy[r]) / 2.0
            
            centers[idx] = [cx, cy]
            radii[idx] = r_cell
            idx += 1
            
    return centers, radii

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
    
    # Try different aspect ratios
    # Perimeter 2(w+h) = 4 => w+h=2
    aspect_ratios = np.linspace(0.5, 1.5, 21)
    
    # Also try square
    aspect_ratios = np.unique(np.concatenate([aspect_ratios, [1.0]]))
    
    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        if h < 0.5: # Ensure reasonable aspect ratio for 7x3 or 3x7
            continue
            
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
        
        # Try 7x3 and 3x7 grids
        for NX, NY in [(7, 3), (3, 7)]:
            if time.time() > deadline - 2:
                break
            try:
                centers, radii = _optimize_grid(w, h, NX, NY, remaining/2.0)
                x = centers[:, 0]
                y = centers[:, 1]
                r = radii
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        # Simple 7x3 uniform grid in 1x1 square
        NX, NY = 7, 3
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            c = i % NX
            r_row = i // NX
            x[i] = (c + 0.5) * (w/NX)
            y[i] = (r_row + 0.5) * (h/NY)
            r[i] = min(w/NX, h/NY) / 2.0
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_result = (x, y, r, w)
            
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
        
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Shrink slightly to ensure validity
    for _ in range(10):
        rt = r * (1 - 1e-6)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
