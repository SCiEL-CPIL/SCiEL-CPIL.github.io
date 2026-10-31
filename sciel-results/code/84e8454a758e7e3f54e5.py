import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 26

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin

    # Define bounds for x, y, r
    # x, y in [0, 1], r in [0, 0.5]
    # To make it easier for DE, we can bound r tighter, e.g., [0.01, 0.3]
    # But circles can be small.
    bounds_x = [(0.0, 1.0) for _ in range(N)]
    bounds_y = [(0.0, 1.0) for _ in range(N)]
    bounds_r = [(0.01, 0.3) for _ in range(N)]
    bounds = bounds_x + bounds_y + bounds_r

    def objective(params):
        """
        Objective function to minimize: -sum(r) + penalty
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # 1. Boundary Penalty
        # Circle i is inside [0,1]^2 if:
        # r_i <= x_i <= 1 - r_i
        # r_i <= y_i <= 1 - r_i
        # Violation: max(0, r_i - x_i) + max(0, x_i - (1 - r_i)) + ...
        
        # Vectorized boundary check
        # Left: x < r -> r - x > 0
        # Right: x > 1 - r -> x - (1 - r) > 0
        # Bottom: y < r -> r - y > 0
        # Top: y > 1 - r -> y - (1 - r) > 0
        
        pen_left = np.maximum(0, r - x)
        pen_right = np.maximum(0, x - (1 - r))
        pen_bottom = np.maximum(0, r - y)
        pen_top = np.maximum(0, y - (1 - r))
        
        bound_pen = np.sum(pen_left**2) + np.sum(pen_right**2) + np.sum(pen_bottom**2) + np.sum(pen_top**2)
        
        # 2. Overlap Penalty
        # d_ij >= r_i + r_j
        # Violation: max(0, r_i + r_j - d_ij)
        
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        
        r_sum = np.add.outer(r, r)
        
        overlap = np.maximum(0, r_sum - d)
        overlap_pen = np.sum(overlap**2)
        
        total_pen = bound_pen + overlap_pen
        
        # Maximize sum(r) -> Minimize -sum(r)
        # Penalty weight needs to be high enough to enforce constraints
        # But not so high that it dominates the gradient in a bad way.
        # Since we are using DE, it's a black-box search, so high penalty is okay.
        lambda_pen = 1000.0
        
        return -np.sum(r) + lambda_pen * total_pen

    # Differential Evolution Settings
    # popsize: 15 * len(bounds) is default, might be too large for 78 vars (1170 individuals).
    # Let's try popsize=10 or 15.
    # maxiter: 100-200 iterations might be enough if popsize is good.
    
    best_val = np.inf
    best_params = None
    
    # Run multiple DE instances with different random seeds to ensure robustness
    for seed in [42, 123, 456]:
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=seed,
                maxiter=200,
                popsize=10,
                tol=1e-5,
                init='latinhypercube',
                polish=True,  # DE does a local polish at the end (Nelder-Mead by default)
                update_interval=10
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                
        except Exception as e:
            pass

    if best_params is None:
        # Fallback: Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Post-processing: Ensure strict validity
    # DE with polish should be good, but let's enforce constraints strictly.
    # If any violation exists, shrink radii.
    
    for _ in range(10):
        # Check boundary
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        # Check overlap
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        overlaps = np.maximum(0, r_sum - d)
        
        max_violation = 0
        if min_x < 0 or max_x > 1 or min_y < 0 or max_y > 1:
            # Calculate max boundary violation
            v = max(-min_x, max_x - 1, -min_y, max_y - 1)
            if v > max_violation:
                max_violation = v
                
        if np.max(overlaps) > max_violation:
            max_violation = np.max(overlaps)
            
        if max_violation < 1e-6:
            break
            
        # Shrink radii to fix violation
        # If we shrink r by delta, overlap reduces by 2*delta (for the pair) and boundary violation by delta.
        # Simple approach: shrink all radii by 1% or proportional to violation.
        r = r * 0.99
        
    # Final clip
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.column_stack((x, y, r))
