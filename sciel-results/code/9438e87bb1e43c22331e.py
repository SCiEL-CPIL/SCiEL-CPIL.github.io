import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 15.0

    # Pre-compute the sofa grid
    # i = 0..449, j = 0..299
    # x_grid = -3 + 0.01 * i
    # y_grid = -1.5 + 0.01 * j
    i = np.arange(450)
    j = np.arange(300)
    X, Y = np.meshgrid(i, j, indexing='ij')
    x_grid = -3 + 0.01 * X
    y_grid = -1.5 + 0.01 * Y
    # Flatten for vectorized ops
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = x_flat.size

    # Pre-compute rotation matrices for all 91 angles
    a_vals = np.arange(91)
    t_vals = a_vals / 90.0
    rad_vals = np.deg2rad(a_vals)
    cos_vals = np.cos(rad_vals)
    sin_vals = np.sin(rad_vals)

    def compute_score(params):
        # params: [cx2, cx3, cy2, cy3]
        cx2, cx3, cy2, cy3 = params
        cx1 = -2 - cx2 - cx3
        cy1 = -cy2 - cy3
        
        # Calculate x, y for each t
        xs = cx1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
        ys = cy1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
        
        # Vectorized check: process all 91 placements at once
        # For each placement k, we need to check if each point is inside
        # We'll build a (91, n_points) boolean array but that's 135k*91 = 12.3M booleans
        # That's ~12MB, which is fine
        
        # For each placement k:
        #   u = (x_flat - xs[k]) * cos[k] + (y_flat - ys[k]) * sin[k]
        #   v = -(x_flat - xs[k]) * sin[k] + (y_flat - ys[k]) * cos[k]
        #   valid[k, p] = (u in [-3,1] and v in [0,1]) or (u in [0,1] and v in [-3,1])
        
        # We can vectorize over k by broadcasting
        # xs has shape (91,), x_flat has shape (n_points,)
        # dx = x_flat[None, :] - xs[:, None]  -> shape (91, n_points)
        # This uses 91 * 135000 * 8 bytes = ~98MB for dx, similar for dy
        # That's ~200MB total, which might be tight but should work
        
        # Alternative: process in chunks of placements to reduce memory
        chunk_size = 10
        mask = np.ones(n_points, dtype=bool)
        
        for start_k in range(0, 91, chunk_size):
            end_k = min(start_k + chunk_size, 91)
            k_slice = slice(start_k, end_k)
            
            xs_chunk = xs[k_slice]  # shape (chunk_size,)
            ys_chunk = ys[k_slice]
            cos_chunk = cos_vals[k_slice]
            sin_chunk = sin_vals[k_slice]
            
            # dx shape: (chunk_size, n_points)
            dx = x_flat[None, :] - xs_chunk[:, None]
            dy = y_flat[None, :] - ys_chunk[:, None]
            
            # u = dx * cos + dy * sin
            u = dx * cos_chunk[:, None] + dy * sin_chunk[:, None]
            v = -dx * sin_chunk[:, None] + dy * cos_chunk[:, None]
            
            # Check base condition for each placement
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid_chunk = cond1 | cond2  # shape (chunk_size, n_points)
            
            # AND over all placements in this chunk
            valid_all = valid_chunk.all(axis=0)  # shape (n_points,)
            mask &= valid_all
            
            # Early exit
            if not np.any(mask):
                return 0.0
        
        count = np.sum(mask)
        return count * 0.0001

    # Objective function for minimization
    def objective(params):
        return -compute_score(params)

    # Initial guesses
    guesses = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.5, 0.0]),
        np.array([-1.0, 0.0, -0.5, 0.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([-0.5, -0.5, -0.5, -0.5]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0, 0.0]),
    ]
    
    best_params = guesses[0].copy()
    best_score = -np.inf
    
    # First, do a coarse grid search to find good starting points
    if time.time() - t_start < time_limit - margin:
        # Coarse grid: 5^4 = 625 evaluations
        # Each eval should be fast now with vectorization
        cx2_vals = np.linspace(-2, 2, 5)
        cx3_vals = np.linspace(-2, 2, 5)
        cy2_vals = np.linspace(-2, 2, 5)
        cy3_vals = np.linspace(-2, 2, 5)
        
        grid_best_score = -np.inf
        grid_best_params = None
        
        for c2 in cx2_vals:
            for c3 in cx3_vals:
                for y2 in cy2_vals:
                    for y3 in cy3_vals:
                        if time.time() - t_start > time_limit - margin:
                            break
                        params = np.array([c2, c3, y2, y3])
                        score = compute_score(params)
                        if score > grid_best_score:
                            grid_best_score = score
                            grid_best_params = params.copy()
                    if time.time() - t_start > time_limit - margin:
                        break
                if time.time() - t_start > time_limit - margin:
                    break
            if time.time() - t_start > time_limit - margin:
                break
        
        if grid_best_params is not None and grid_best_score > best_score:
            best_score = grid_best_score
            best_params = grid_best_params.copy()
        
        # Add grid best to guesses
        if grid_best_params is not None:
            guesses.append(grid_best_params.copy())
    
    # Now run Nelder-Mead from multiple starting points
    for guess in guesses:
        if time.time() - t_start > time_limit - margin:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If we still have time, do a finer local search around the best
    if time.time() - t_start < time_limit - margin and best_score > 0:
        # Try small perturbations around best_params
        for dim in range(4):
            for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                if time.time() - t_start > time_limit - margin:
                    break
                trial = best_params.copy()
                trial[dim] += delta
                try:
                    score = compute_score(trial)
                    if score > best_score:
                        best_score = score
                        best_params = trial.copy()
                except Exception:
                    pass
    
    # Construct the final path from best_params
    cx2, cx3, cy2, cy3 = best_params
    cx1 = -2 - cx2 - cx3
    cy1 = -cy2 - cy3

    xs = cx1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
    ys = cy1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
    
    path_pts = np.column_stack((xs, ys, a_vals))
    
    return path_pts
