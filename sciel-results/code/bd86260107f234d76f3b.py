import numpy as np
import time
from scipy.optimize import minimize

def compute_area_exact(c, X, Y, N=8):
    """
    Compute the area of the union of N*N rectangles at height c.
    Key insight: cell (x-slab, y-slab) is covered iff
    (exists i with x-interval covering x-slab) AND (exists j with y-interval covering y-slab)
    because all (i,j) combinations exist.
    X: array of shape (N,)
    Y: array of shape (N,)
    """
    i = np.arange(N)
    # x-intervals: for each i, [X[i] + c*i/N, X[i] + c*i/N + 1/N]
    x_left_i = X + c * i / N
    x_right_i = x_left_i + 1.0 / N
    # y-intervals: for each j, [Y[j] + c*j/N, Y[j] + c*j/N + 1/N]
    j = np.arange(N)
    y_left_j = Y + c * j / N
    y_right_j = y_left_j + 1.0 / N
    
    # Get unique edges
    x_edges = np.unique(np.concatenate([x_left_i, x_right_i]))
    y_edges = np.unique(np.concatenate([y_left_j, y_right_j]))
    
    area = 0.0
    # Iterate over cells in the grid defined by edges
    for px in range(len(x_edges) - 1):
        x0 = x_edges[px]
        x1 = x_edges[px + 1]
        if x1 <= x0:
            continue
        # Check which i's cover this x-slab
        cover_i = (x_left_i <= x0) & (x_right_i >= x1)
        if not np.any(cover_i):
            continue
        
        for py in range(len(y_edges) - 1):
            y0 = y_edges[py]
            y1 = y_edges[py + 1]
            if y1 <= y0:
                continue
            # Check which j's cover this y-slab
            cover_j = (y_left_j <= y0) & (y_right_j >= y1)
            if not np.any(cover_j):
                continue
            
            # Cell is covered if there exists at least one i covering x-slab
            # AND at least one j covering y-slab
            # Since all (i,j) combinations exist, this is sufficient
            area += (x1 - x0) * (y1 - y0)
                
    return area

def compute_volume(X, Y, N=8, num_c=30):
    """
    Estimate volume by integrating area over c using Gauss-Legendre quadrature.
    """
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
    max_time = 620.0  # Safety margin before 630s kill
    
    # Initial guesses
    X_init = np.zeros(N)
    Y_init = np.zeros(N)
    
    best_vol = compute_volume(X_init, Y_init)
    best_params = np.concatenate([X_init, Y_init])
    
    rng = np.random.default_rng(42)
    
    # Define objective function for optimizer
    def objective(params):
        if time.time() - start_time > max_time - 5:
            return 1e10  # Penalty for timeout
        X = params[:N]
        Y = params[N:]
        try:
            return compute_volume(X, Y)
        except:
            return 1e10

    # Generate diverse starting points
    starts = [
        np.concatenate([np.zeros(N), np.zeros(N)]),
        np.concatenate([-np.arange(N) / (2 * N), -np.arange(N) / (2 * N)]),
        np.concatenate([np.arange(N) / (2 * N), np.arange(N) / (2 * N)]),
        np.concatenate([-np.arange(N) / N, np.arange(N) / N]),
        np.concatenate([np.arange(N) / N, -np.arange(N) / N]),
    ]
    
    # Add random starts
    for _ in range(8):
        X_r = rng.uniform(-0.5, 0.5, N)
        Y_r = rng.uniform(-0.5, 0.5, N)
        starts.append(np.concatenate([X_r, Y_r]))
    
    # Run Nelder-Mead from multiple starts
    for start in starts:
        if time.time() - start_time > max_time - 10:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except:
            pass
            
    # Run Powell from best few starts
    for start in starts[:5]:
        if time.time() - start_time > max_time - 10:
            break
        try:
            res = minimize(objective, start, method='Powell',
                           options={'maxiter': 800, 'xtol': 1e-5, 'ftol': 1e-7})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except:
            pass

    # Final local polish with coordinate descent if time permits
    if time.time() - start_time < max_time - 30:
        X = best_params[:N].copy()
        Y = best_params[N:].copy()
        curr_vol = best_vol
        step = 0.005
        for _ in range(300):
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
                if step < 1e-6:
                    break
            else:
                step = min(step * 1.05, 0.02)
        
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
