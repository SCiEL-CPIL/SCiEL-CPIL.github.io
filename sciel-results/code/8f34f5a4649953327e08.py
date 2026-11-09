import numpy as np
import time
from scipy.optimize import minimize

def compute_area_exact(c, X, Y, N=8):
    """
    Compute the area of the union of N*N rectangles at height c.
    X: array of shape (N,)
    Y: array of shape (N,)
    """
    # Generate x-intervals and y-intervals for all N*N tubes
    # Tube (i, j)
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    x_left = X[i_idx] + c * i_idx / N
    x_right = x_left + 1.0 / N
    y_left = Y[j_idx] + c * j_idx / N
    y_right = y_left + 1.0 / N
    
    # Get unique edges
    x_edges = np.unique(np.concatenate([x_left, x_right]))
    y_edges = np.unique(np.concatenate([y_left, y_right]))
    
    area = 0.0
    # Iterate over cells in the grid defined by edges
    # To speed up, we can vectorize the check
    # Create meshgrid of cell centers
    # Or just iterate, N=8 means max 16 edges per dim, 15x15=225 cells.
    # 225 cells * 64 tubes = 14400 checks. Very fast.
    
    for px in range(len(x_edges) - 1):
        x0 = x_edges[px]
        x1 = x_edges[px + 1]
        if x1 <= x0:
            continue
        # Check which tubes cover this x-slice
        # A tube covers [x0, x1] if x_left <= x0 and x_right >= x1
        # Since x0, x1 are edges, this is exact
        cover_i_mask = (x_left <= x0) & (x_right >= x1)
        if not np.any(cover_i_mask):
            continue
        
        for py in range(len(y_edges) - 1):
            y0 = y_edges[py]
            y1 = y_edges[py + 1]
            if y1 <= y0:
                continue
            cover_j_mask = (y_left <= y0) & (y_right >= y1)
            if not np.any(cover_j_mask):
                continue
            
            # If there is at least one tube covering x-slice and at least one covering y-slice,
            # it does NOT mean the cell is covered.
            # We need a tube (i,j) such that it covers BOTH x-slice and y-slice.
            # The x-slice is covered by tubes with indices in `cover_i_mask`
            # The y-slice is covered by tubes with indices in `cover_j_mask`
            # We need intersection of these sets of tube indices.
            
            # cover_i_mask is boolean array of length N*N
            # cover_j_mask is boolean array of length N*N
            # If any tube satisfies both, the cell is covered.
            if np.any(cover_i_mask & cover_j_mask):
                area += (x1 - x0) * (y1 - y0)
                
    return area

def compute_volume(X, Y, N=8, num_c=20):
    """
    Estimate volume by integrating area over c.
    """
    # Use Gauss-Legendre quadrature for better accuracy with fewer points
    # or just simple midpoint rule which is robust
    c_vals, w_vals = np.polynomial.legendre.leggauss(num_c)
    # Map from [-1, 1] to [0, 1]
    c_vals = (c_vals + 1) / 2
    w_vals = w_vals / 2
    
    total_vol = 0.0
    for c, w in zip(c_vals, w_vals):
        area = compute_area_exact(c, X, Y, N)
        total_vol += w * area
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    max_time = 590.0  # Safety margin
    
    # Parameters: X (N), Y (N)
    # Initial guess: Counter-shearing
    # X_i = -i / (2*N) ? 
    # Let's try X_i = 0, Y_j = 0 first as baseline
    X_init = np.zeros(N)
    Y_init = np.zeros(N)
    
    best_vol = compute_volume(X_init, Y_init)
    best_params = np.concatenate([X_init, Y_init])
    
    rng = np.random.default_rng(42)
    
    # Define objective function for optimizer
    def objective(params):
        if time.time() - start_time > max_time - 5:
            return 1e10 # Penalty for timeout
        X = params[:N]
        Y = params[N:]
        try:
            return compute_volume(X, Y)
        except:
            return 1e10

    # Run multiple optimizations from different starts
    # 1. Zero start
    # 2. Counter-shear start: X_i = -i/(2*N), Y_j = -j/(2*N)
    # 3. Random starts
    
    starts = [
        np.concatenate([np.zeros(N), np.zeros(N)]),
        np.concatenate([-np.arange(N)/(2*N), -np.arange(N)/(2*N)]),
        np.concatenate([np.arange(N)/(2*N), np.arange(N)/(2*N)]),
    ]
    
    for _ in range(5):
        X_r = rng.uniform(-0.5, 0.5, N)
        Y_r = rng.uniform(-0.5, 0.5, N)
        starts.append(np.concatenate([X_r, Y_r]))
    
    for start in starts:
        if time.time() - start_time > max_time - 10:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except:
            pass
            
    # Also try Powell which can handle non-smoothness better
    for start in starts[:3]:
        if time.time() - start_time > max_time - 10:
            break
        try:
            res = minimize(objective, start, method='Powell',
                           options={'maxiter': 500, 'xtol': 1e-4, 'ftol': 1e-6})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except:
            pass

    # Final local polish with coordinate descent if time permits
    if time.time() - start_time < max_time - 20:
        X = best_params[:N].copy()
        Y = best_params[N:].copy()
        curr_vol = best_vol
        step = 0.01
        for _ in range(200):
            if time.time() - start_time > max_time - 5:
                break
            improved = False
            for i in range(N):
                for sign in [-1, 1]:
                    X_new = X.copy()
                    X_new[i] += sign * step
                    v = compute_volume(X_new, Y)
                    if v < curr_vol:
                        X = X_new
                        curr_vol = v
                        improved = True
                        break
            for j in range(N):
                for sign in [-1, 1]:
                    Y_new = Y.copy()
                    Y_new[j] += sign * step
                    v = compute_volume(X, Y_new)
                    if v < curr_vol:
                        Y = Y_new
                        curr_vol = v
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
            else:
                step = min(step * 1.1, 0.05)
        
        if curr_vol < best_vol:
            best_vol = curr_vol
            best_params = np.concatenate([X, Y])

    X_final = best_params[:N]
    Y_final = best_params[N:]
    
    # Construct the 64x2 array
    # k = i * 8 + j
    # x_k = X_i, y_k = Y_j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_final[i_flat]
    base_y = Y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
