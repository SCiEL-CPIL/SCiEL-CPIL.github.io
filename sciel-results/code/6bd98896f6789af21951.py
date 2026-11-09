import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 10.0

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

    def compute_score(params):
        # params: [cx2, cx3, cy2, cy3]
        # x(t) = cx1*t + cx2*t^2 + cx3*t^3
        # y(t) = cy1*t + cy2*t^2 + cy3*t^3
        # x(0)=0, y(0)=0
        # x(1) = cx1 + cx2 + cx3 = -2  => cx1 = -2 - cx2 - cx3
        # y(1) = cy1 + cy2 + cy3 = 0   => cy1 = -cy2 - cy3
        
        cx2, cx3, cy2, cy3 = params
        cx1 = -2 - cx2 - cx3
        cy1 = -cy2 - cy3

        # We will output points at every integer degree a = 0, 1, ..., 90
        # t = a / 90
        a_vals = np.arange(91)
        t_vals = a_vals / 90.0
        
        # Calculate x, y for each t
        # x(t) = cx1*t + cx2*t^2 + cx3*t^3
        xs = cx1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
        ys = cy1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
        
        # Construct path points (x, y, a)
        # The evaluator will interpolate between these.
        # Since step is 1 degree, m=1, no extra points.
        path_pts = np.column_stack((xs, ys, a_vals))
        
        # Check all placements
        # Start (0,0,0) is included in path_pts (t=0 => x=0, y=0, a=0)
        # End (-2,0,90) is included in path_pts (t=1 => x=-2, y=0, a=90)
        
        # Vectorized check for all placements
        # Placements: (xs, ys, a_vals)
        # Sofa points: (x_flat, y_flat)
        
        # We need to check if each sofa point is inside ALL placements.
        # This is a min over placements of (1 if inside else 0).
        # Or AND over all placements.
        
        # To avoid huge memory usage (135k * 91), we can loop over placements
        # and update a mask.
        mask = np.ones(n_points, dtype=bool)
        
        for k in range(len(a_vals)):
            cx = xs[k]
            cy = ys[k]
            ca = a_vals[k]
            
            rad = np.deg2rad(ca)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            
            # Translate
            dx = x_flat - cx
            dy = y_flat - cy
            
            # Rotate by -alpha
            # R(-a) = [[cos, sin], [-sin, cos]]
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            # Check base condition
            # Base: (u in [-3, 1] and v in [0, 1]) or (u in [0, 1] and v in [-3, 1])
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid = cond1 | cond2
            
            mask &= valid
            
            # Early exit if no points remain
            if not np.any(mask):
                return 0.0
        
        count = np.sum(mask)
        return count * 0.0001

    # Objective function for minimization (we want to maximize score, so minimize -score)
    def objective(params):
        return -compute_score(params)

    # Initial guess
    # Linear path: cx2=0, cx3=0, cy2=0, cy3=0
    # x(t) = -2t, y(t) = 0
    # This is the seed path essentially.
    initial_guess = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Let's try a few initial guesses to avoid local minima
    guesses = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([-0.5, -0.5, -0.5, -0.5]),
    ]
    
    best_params = initial_guess
    best_score = -np.inf
    
    for guess in guesses:
        if time.time() - t_start > time_limit - margin:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass

    # If optimization failed or didn't improve, use the best grid point or initial guess
    # Let's also do a quick grid search if we have time
    if time.time() - t_start < time_limit - margin:
        # Coarse grid search
        cx2_vals = np.linspace(-2, 2, 11)
        cx3_vals = np.linspace(-2, 2, 11)
        cy2_vals = np.linspace(-2, 2, 11)
        cy3_vals = np.linspace(-2, 2, 11)
        
        # This is 11^4 = 14641 evaluations. Might be too slow if each takes 10ms.
        # 14641 * 0.01 = 146s. Doable if optimized.
        # But let's just pick a few better starting points for Nelder-Mead instead.
        
        # Let's just run Nelder-Mead on the best found so far with tighter bounds?
        # No, let's just use the best_params found.

    # Construct the final path from best_params
    cx2, cx3, cy2, cy3 = best_params
    cx1 = -2 - cx2 - cx3
    cy1 = -cy2 - cy3

    a_vals = np.arange(91)
    t_vals = a_vals / 90.0
    
    xs = cx1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
    ys = cy1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
    
    path_pts = np.column_stack((xs, ys, a_vals))
    
    return path_pts
