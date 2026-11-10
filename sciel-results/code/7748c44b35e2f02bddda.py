# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_2d_union_area(x, y, c, N=8, res=32):
    """Compute the 2D union area of N^2 squares at height c.
    x, y: arrays of shape (N,)
    c: scalar height
    N: number of tubes per dimension
    res: grid resolution for area estimation
    """
    i_idx = np.arange(N)
    
    # Square (i,j) is [x_i + c*i/N, x_i + (i+1)/N + c*i/N] x [y_j + c*j/N, y_j + (j+1)/N + c*j/N]
    
    # Calculate min and max x, y for all squares to define bounding box
    x_starts = x + c * i_idx / N
    x_ends = x_starts + 1.0 / N
    y_starts = y + c * i_idx / N
    y_ends = y_starts + 1.0 / N
    
    x_min = np.min(x_starts)
    x_max = np.max(x_ends)
    y_min = np.min(y_starts)
    y_max = np.max(y_ends)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 1e-10 or height <= 1e-10:
        return 0.0
        
    # Grid
    grid = np.zeros((res, res), dtype=np.int8)
    
    # Mark each square on the grid
    for i in range(N):
        for j in range(N):
            sx = x[i] + c * i / N
            ex = sx + 1.0 / N
            sy = y[j] + c * j / N
            ey = sy + 1.0 / N
            
            # Map to grid indices
            # x_grid = (x - x_min) / width * res
            x_start_idx = int((sx - x_min) / width * res)
            x_end_idx = int((ex - x_min) / width * res)
            y_start_idx = int((sy - y_min) / height * res)
            y_end_idx = int((ey - y_min) / height * res)
            
            # Ensure bounds
            x_start_idx = max(0, min(res, x_start_idx))
            x_end_idx = max(0, min(res, x_end_idx))
            y_start_idx = max(0, min(res, y_start_idx))
            y_end_idx = max(0, min(res, y_end_idx))
            
            if x_start_idx < x_end_idx and y_start_idx < y_end_idx:
                grid[y_start_idx:y_end_idx, x_start_idx:x_end_idx] = 1
                
    area = np.sum(grid) * (width / res) * (height / res)
    return area

def _compute_3d_volume(x, y, N=8, res=32, K=50):
    """Compute the 3D union volume by integrating 2D union area over c in [0, 1]."""
    total_vol = 0.0
    for k in range(K):
        c = k / K
        total_vol += _compute_2d_union_area(x, y, c, N, res)
    return total_vol / K

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0

    # Initial candidates for x and y coordinates
    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Symmetric
    symmetric = -np.arange(N, dtype=np.float64) / (2 * N)
    
    # Zero
    origin = np.zeros(N, dtype=np.float64)
    
    # Linear
    linear = np.linspace(0, 1 - 1 / N, N, dtype=np.float64)
    
    candidates = [keich, symmetric, origin, linear]
    
    best_params = np.concatenate([keich, keich])
    best_vol = float('inf')
    
    # Evaluate initial candidates
    for x in candidates:
        for y in candidates:
            vol = _compute_3d_volume(x, y, N, res=32, K=20)
            if vol < best_vol:
                best_vol = vol
                best_params = np.concatenate([x, y])
                
    # Optimization
    if time.time() < deadline - 60.0:
        from scipy.optimize import minimize
        
        def objective(params):
            x = params[:N]
            y = params[N:]
            return _compute_3d_volume(x, y, N, res=32, K=50)
            
        x0 = best_params.copy()
        
        # Powell method is good for this
        try:
            res = minimize(objective, x0, method='Powell',
                          options={'maxiter': 500, 'ftol': 1e-8})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
        # Try Nelder-Mead if time remains
        if time.time() < deadline - 30.0:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res2.fun < best_vol:
                    best_vol = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    best_x = best_params[:N]
    best_y = best_params[N:]
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions

# EVOLVE-BLOCK-END
