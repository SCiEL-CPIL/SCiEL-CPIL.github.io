import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 20.0

    # Pre-compute the sofa grid
    i_arr = np.arange(450)
    j_arr = np.arange(300)
    I, J = np.meshgrid(i_arr, j_arr, indexing='ij')
    x_grid = -3 + 0.01 * I
    y_grid = -1.5 + 0.01 * J
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = x_flat.size  # 135000

    # Pre-compute rotations for 91 angles
    a_vals = np.arange(91)
    t_vals = a_vals / 90.0
    rad_vals = np.deg2rad(a_vals)
    cos_vals = np.cos(rad_vals)
    sin_vals = np.sin(rad_vals)

    def compute_score_from_xy(xs, ys):
        """xs, ys: arrays of length 91, positions at each angle"""
        mask = np.ones(n_points, dtype=bool)
        
        for k in range(91):
            cx = xs[k]
            cy = ys[k]
            ca = cos_vals[k]
            sa = sin_vals[k]
            
            dx = x_flat - cx
            dy = y_flat - cy
            
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid = cond1 | cond2
            
            mask &= valid
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001

    def compute_score_from_params(params):
        """params: [cx2, cx3, cy2, cy3]
        x(t) = cx1*t + cx2*t^2 + cx3*t^3
        y(t) = cy1*t + cy2*t^2 + cy3*t^3
        where cx1 = -2 - cx2 - cx3, cy1 = -cy2 - cy3
        so x(0)=0, x(1)=-2, y(0)=0, y(1)=0
        """
        cx2, cx3, cy2, cy3 = params
        cx1 = -2 - cx2 - cx3
        cy1 = -cy2 - cy3
        
        xs = cx1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
        ys = cy1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
        
        return compute_score_from_xy(xs, ys)

    # Objective function for minimization
    def objective(params):
        return -compute_score_from_params(params)

    # Initial guesses - various polynomial paths
    guesses = [
        np.array([0.0, 0.0, 0.0, 0.0]),       # linear
        np.array([1.0, 0.0, 0.5, 0.0]),
        np.array([-1.0, 0.0, -0.5, 0.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([-0.5, -0.5, -0.5, -0.5]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0, 0.0]),
        np.array([1.5, -0.5, 0.8, -0.3]),
        np.array([-1.5, 0.5, -0.8, 0.3]),
        np.array([2.0, -1.0, 1.2, -0.5]),
        np.array([-2.0, 1.0, -1.2, 0.5]),
        np.array([0.8, -0.3, 1.5, -0.7]),
        np.array([-0.8, 0.3, -1.5, 0.7]),
    ]
    
    best_params = guesses[0].copy()
    best_score = -np.inf
    
    # First, evaluate all initial guesses
    for guess in guesses:
        if time.time() - t_start > time_limit - margin:
            break
        score = compute_score_from_params(guess)
        if score > best_score:
            best_score = score
            best_params = guess.copy()
    
    # Coarse grid search to find good starting points
    if time.time() - t_start < time_limit - margin:
        # Coarse grid: 4^4 = 256 evaluations
        cx2_vals = np.linspace(-3, 3, 4)
        cx3_vals = np.linspace(-3, 3, 4)
        cy2_vals = np.linspace(-3, 3, 4)
        cy3_vals = np.linspace(-3, 3, 4)
        
        grid_best_score = -np.inf
        grid_best_params = None
        
        for c2 in cx2_vals:
            for c3 in cx3_vals:
                for y2 in cy2_vals:
                    for y3 in cy3_vals:
                        if time.time() - t_start > time_limit - margin:
                            break
                        params = np.array([c2, c3, y2, y3])
                        score = compute_score_from_params(params)
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
        
        # Add grid best to guesses if found
        if grid_best_params is not None:
            guesses.append(grid_best_params.copy())
    
    # Run Nelder-Mead from multiple starting points
    for guess in guesses:
        if time.time() - t_start > time_limit - margin:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
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
            for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5]:
                if time.time() - t_start > time_limit - margin:
                    break
                trial = best_params.copy()
                trial[dim] += delta
                try:
                    score = compute_score_from_params(trial)
                    if score > best_score:
                        best_score = score
                        best_params = trial.copy()
                except Exception:
                    pass
        
        # Another round of Nelder-Mead from the improved point
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9,
                                    'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Construct the final path from best_params
    cx2, cx3, cy2, cy3 = best_params
    cx1 = -2 - cx2 - cx3
    cy1 = -cy2 - cy3

    xs = cx1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
    ys = cy1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
    
    path_pts = np.column_stack((xs, ys, a_vals))
    
    # Ensure finite
    if not np.all(np.isfinite(path_pts)):
        # Fallback to linear path
        path_pts = np.column_stack((np.linspace(0, -2, 91), np.zeros(91), a_vals))
    
    # Final verification
    final_score = compute_score_from_xy(path_pts[:, 0], path_pts[:, 1])
    
    return path_pts
