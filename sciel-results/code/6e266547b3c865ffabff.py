# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# Pre-compute grid points for efficiency
_XS = np.arange(-3, 1.5, 0.01)
_YS = np.arange(-1.5, 1.5, 0.01)
_XG, _YG = np.meshgrid(_XS, _YS)
_POINTS = np.vstack([_XG.ravel(), _YG.ravel()]).T
_N_POINTS = _POINTS.shape[0]

def _calculate_area_from_densified_path(densified_path):
    """
    Calculate the area of the intersection of the corridor placements
    given a fully densified path (1 degree steps).
    """
    if len(densified_path) == 0:
        return 0.0
    
    mask = np.ones(_N_POINTS, dtype=bool)
    
    for p in densified_path:
        x, y, a = p
        a_rad = np.deg2rad(a)
        c = np.cos(-a_rad)
        s = np.sin(-a_rad)
        
        # Translate
        dx = _POINTS[:, 0] - x
        dy = _POINTS[:, 1] - y
        
        # Rotate
        u = c * dx - s * dy
        v = s * dx + c * dy
        
        # Check base corridor:
        # Part 1: 0 <= u <= 1, -3 <= v <= 1
        # Part 2: -3 <= u <= 1, 0 <= v <= 1
        m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        mask &= (m1 | m2)
        
        if not np.any(mask):
            return 0.0
            
    return np.sum(mask) * 0.0001

def _densify_path(path):
    """
    Densify the path to 1-degree steps as per the evaluator's logic.
    Input: list of (x, y, angle)
    Output: np.array of densified (x, y, angle)
    """
    if len(path) < 2:
        return path
    
    # Ensure sorted by angle
    path = np.array(path)
    path = path[path[:, 2].argsort()]
    
    densified = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        a1 = p1[2]
        a2 = p2[2]
        
        # m = max(1, ceil(|a2 - a1|))
        # The evaluator adds interpolations in m equal steps.
        # This implies m segments, so m-1 intermediate points?
        # "adds linear interpolations ... in m equal steps"
        # If I go from 0 to 10 in 10 steps, I get 1, 2, ..., 9.
        # So the step size is (a2-a1)/m.
        m = max(1, int(np.ceil(abs(a2 - a1))))
        
        if m == 1:
            densified.append(p1)
        else:
            # Generate m-1 intermediate points?
            # Or does "in m steps" mean the resulting segment count is m?
            # If I divide a segment into m steps, I have m sub-segments.
            # So I need points at t = 1/m, 2/m, ..., (m-1)/m.
            # The start point p1 is already added in the previous iteration (or initially).
            # The end point p2 will be added in the next iteration.
            
            for k in range(1, m):
                t = k / m
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t
                a = a1 * (1 - t) + a2 * t
                densified.append([x, y, a])
    # Add the last point
    densified.append(path[-1])
    
    return np.array(densified)

def _evaluate_path(path_rows):
    """
    Evaluate the objective for a set of path rows (excluding start and end).
    """
    # Add start and end
    full_path = np.vstack([
        [[0.0, 0.0, 0.0]],
        path_rows,
        [[-2.0, 0.0, 90.0]]
    ])
    
    # Sort by angle
    full_path = full_path[full_path[:, 2].argsort()]
    
    # Densify
    densified = _densify_path(full_path)
    
    # Calculate Area
    area = _calculate_area_from_densified_path(densified)
    return area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess based on the seed program
    seed_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    # We will optimize points at 10, 20, 30, 40, 50, 60, 70, 80 degrees
    target_angles = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    
    # Interpolate seed to get initial guess
    init_x = np.interp(target_angles, seed_path[:, 2], seed_path[:, 0])
    init_y = np.interp(target_angles, seed_path[:, 2], seed_path[:, 1])
    
    init_params = np.concatenate([init_x, init_y])
    
    def objective(params):
        x = params[:8]
        y = params[8:]
        path = np.column_stack([x, y, target_angles])
        return -_evaluate_path(path) # Minimize negative area
        
    # Bounds
    bounds = []
    for i in range(8):
        bounds.append((-2.0, 0.5)) # x
    for i in range(8):
        bounds.append((-0.5, 1.0)) # y
        
    best_params = init_params
    best_score = _evaluate_path(np.column_stack([init_x, init_y, target_angles]))
    
    # Strategy: 
    # 1. Differential Evolution for global search (short run)
    # 2. Nelder-Mead for local polish
    
    try:
        # Global Search
        # Use a small popsize and few iterations to stay within time
        result_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=5, 
            popsize=5, 
            tol=0.1,
            polish=False
        )
        
        if result_de.fun < objective(best_params):
            best_params = result_de.x
            best_score = -result_de.fun
            
    except Exception as e:
        pass
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 60: # Leave 60s for final polish and safety
        
        # Local Polish
        try:
            result_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if result_nm.fun < objective(best_params):
                best_params = result_nm.x
                best_score = -result_nm.fun
        except Exception as e:
            pass

    # Construct final path
    best_x = best_params[:8]
    best_y = best_params[8:]
    final_path = np.column_stack([best_x, best_y, target_angles])
    
    # Final check: ensure it's valid and finite
    if not np.all(np.isfinite(final_path)):
        return seed_path
        
    return final_path

# EVOLVE-BLOCK-END
