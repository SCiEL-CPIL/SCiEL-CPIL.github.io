"""Circle packing in perimeter-4 rectangle: Optimized Grid-based Cellular Decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # Define grid configurations
    # (nx, ny, active_indices)
    configs = []
    
    # 5x5 grid, drop 4 corners
    nx, ny = 5, 5
    active_5x5 = []
    for r in range(ny):
        for c in range(nx):
            if (r == 0 and c == 0) or (r == 0 and c == nx-1) or (r == ny-1 and c == 0) or (r == ny-1 and c == nx-1):
                continue
            active_5x5.append(r * nx + c)
    configs.append((nx, ny, np.array(active_5x5)))
    
    # 3x7 grid, all active
    nx, ny = 3, 7
    active_3x7 = np.arange(nx * ny)
    configs.append((nx, ny, active_3x7))
    
    # 7x3 grid, all active
    nx, ny = 7, 3
    active_7x3 = np.arange(nx * ny)
    configs.append((nx, ny, active_7x3))
    
    # 4x6 grid, drop 3 corners
    nx, ny = 4, 6
    active_4x6 = []
    for r in range(ny):
        for c in range(nx):
            if (r == 0 and c == nx-1) or (r == ny-1 and c == 0) or (r == ny-1 and c == nx-1):
                continue
            active_4x6.append(r * nx + c)
    configs.append((nx, ny, np.array(active_4x6)))
    
    # 6x4 grid, drop 3 corners
    nx, ny = 6, 4
    active_6x4 = []
    for r in range(ny):
        for c in range(nx):
            if (r == 0 and c == nx-1) or (r == ny-1 and c == 0) or (r == ny-1 and c == nx-1):
                continue
            active_6x4.append(r * nx + c)
    configs.append((nx, ny, np.array(active_6x4)))
    
    # 2x11 grid
    nx, ny = 2, 11
    active_2x11 = np.arange(nx * ny)
    configs.append((nx, ny, active_2x11))
    
    # 11x2 grid
    nx, ny = 11, 2
    active_11x2 = np.arange(nx * ny)
    configs.append((nx, ny, active_11x2))

    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.5, 0.67]
    
    for nx, ny, active_idx in configs:
        if time.time() > deadline - 5:
            break
            
        for w_init in aspect_ratios:
            if time.time() > deadline - 2:
                break
                
            h_init = 2.0 - w_init
            if h_init < 0.2:
                continue
                
            # Variables: alpha (nx), beta (ny)
            # Sum(alpha) + Sum(beta) = 2
            alpha_init = np.full(nx, w_init / nx)
            beta_init = np.full(ny, h_init / ny)
            x0 = np.concatenate([alpha_init, beta_init])
            
            bounds = [(1e-4, 2.0)] * (nx + ny)
            
            def objective(theta):
                alpha = theta[:nx]
                beta = theta[nx:nx+ny]
                
                radii = np.zeros(len(active_idx))
                for i, idx in enumerate(active_idx):
                    r = idx // nx
                    c = idx % nx
                    radii[i] = min(alpha[c], beta[r]) / 2.0
                
                return -np.sum(radii)
            
            constraints = [{'type': 'eq', 'fun': lambda theta: np.sum(theta) - 2.0}]
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
                
                if res.success:
                    theta_opt = res.x
                    alpha_opt = theta_opt[:nx]
                    beta_opt = theta_opt[nx:nx+ny]
                    
                    x_centers = np.zeros(len(active_idx))
                    y_centers = np.zeros(len(active_idx))
                    radii = np.zeros(len(active_idx))
                    
                    alpha_cum = np.cumsum(np.insert(alpha_opt, 0, 0))
                    beta_cum = np.cumsum(np.insert(beta_opt, 0, 0))
                    
                    for i, idx in enumerate(active_idx):
                        r = idx // nx
                        c = idx % nx
                        
                        x_left = alpha_cum[c]
                        x_right = alpha_cum[c+1]
                        y_bottom = beta_cum[r]
                        y_top = beta_cum[r+1]
                        
                        x_centers[i] = (x_left + x_right) / 2.0
                        y_centers[i] = (y_bottom + y_top) / 2.0
                        radii[i] = min(alpha_opt[c], beta_opt[r]) / 2.0
                    
                    w_opt = np.sum(alpha_opt)
                    h_opt = np.sum(beta_opt)
                    
                    valid, val = _check_validity(x_centers, y_centers, radii, w_opt, h_opt)
                    if valid and val > best_value:
                        best_value = val
                        best_result = (x_centers.copy(), y_centers.copy(), radii.copy(), w_opt)
                        
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
    
    for _ in range(5):
        rt = r * (1 - 1e-8)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))

def _check_validity(x, y, r, w, h, tol=1e-9):
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    for i in range(len(r)):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False, 0.0
    return True, float(np.sum(r))
