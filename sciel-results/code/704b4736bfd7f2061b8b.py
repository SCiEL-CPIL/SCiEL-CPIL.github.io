# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _build_grid(step=1):
    """Build the grid for the sofa points."""
    dx = 0.01 * step
    xs = np.arange(-3.0, 1.5, dx)
    ys = np.arange(-1.5, 1.5, dx)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _evaluate_path_fast(path, grid_x, grid_y):
    """
    Fast evaluation: count points inside all placements.
    Returns raw value (count * 0.0001).
    """
    dense = _densify_path(path)
    n_points = len(grid_x)
    inside = np.ones(n_points, dtype=bool)
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = grid_x - x
        dy = grid_y - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside &= in_base
        
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _densify_path(path):
    """
    Densify the path by adding linear interpolations between consecutive waypoints.
    The evaluator adds (0,0,0) at start and (-2,0,90) at end.
    Interpolation steps m = max(1, ceil(|a2 - a1|)).
    """
    # Prepend start and append end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Ensure path angles are continuous/unwrapped for interpolation logic consistency
    # The evaluator uses linear interpolation of (x,y,a).
    # If a jumps from 0 to 350, it might interpolate the long way? 
    # The task says "linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps".
    # It implies a is treated as a scalar value.
    
    full = np.vstack([start, path, end])
    
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(math.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)

def _evaluate_path(path, grid_x, grid_y):
    """
    Evaluate the area of the sofa that fits through the path defined by `path`.
    Returns the raw value (number of points * 0.0001).
    """
    dense = _densify_path(path)
    n_points = len(grid_x)
    
    # Initialize all points as 'inside'
    inside = np.ones(n_points, dtype=bool)
    
    # Pre-allocate arrays for rotated coords to save memory/alloc time?
    # No, just compute on the fly for each placement.
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Rotate grid points relative to (x,y)
        dx = grid_x - x
        dy = grid_y - y
        
        # R(-a) (w - (x,y))
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Check base corridor:
        # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside &= in_base
        
        # Early exit if no points are inside
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _nelder_mead_optimize(params, grid_x, grid_y, deadline, initial_step=0.5, max_iter=200, eval_func=None):
    """
    Custom Nelder-Mead implementation to allow better control and early termination.
    Objective: Maximize eval_func.
    """
    if eval_func is None:
        eval_func = _evaluate_path_fast
    
    n = len(params)
    # Build simplex
    simplex = [params.copy()]
    for i in range(n):
        p = params.copy()
        p[i] += initial_step
        simplex.append(p)
        
    # Evaluate initial simplex
    f_values = [eval_func(p.reshape(-1, 3), grid_x, grid_y) for p in simplex]
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort
        order = np.argsort(f_values)[::-1] # Descending
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Check convergence
        if abs(f_values[0] - f_values[-1]) < 1e-8:
            break
            
        # Compute centroid of all but the worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - worst)
        f_r = eval_func(x_r.reshape(-1, 3), grid_x, grid_y)
        
        if f_r > f_values[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (centroid - worst)
            f_e = eval_func(x_e.reshape(-1, 3), grid_x, grid_y)
            if f_e > f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r >= f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            # Contraction
            beta = 0.5
            x_c = centroid + beta * (worst - centroid)
            f_c = eval_func(x_c.reshape(-1, 3), grid_x, grid_y)
            if f_c > f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_values[i] = eval_func(simplex[i].reshape(-1, 3), grid_x, grid_y)
        
    # Find best in simplex
    best_idx = np.argmax(f_values)
    return simplex[best_idx], f_values[best_idx]

def _coordinate_descent(params, grid_x, grid_y, deadline, step_sizes=None, eval_func=None):
    """
    Coordinate descent for fine-tuning on a specific grid.
    """
    if eval_func is None:
        eval_func = _evaluate_path_fast
    
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
        
    n = len(params) // 3
    path = params.copy().reshape(n, 3)
    best_area = eval_func(path, grid_x, grid_y)
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for dim in range(3):
                    for direction in [1, -1]:
                        trial = path.copy()
                        trial[i, dim] += direction * step_size
                        # Clip angles to reasonable range to avoid unnecessary wrap-around issues
                        if dim == 2:
                            trial[i, dim] = np.clip(trial[i, dim], -90, 180)
                        
                        area = eval_func(trial, grid_x, grid_y)
                        if area > best_area:
                            path = trial
                            best_area = area
                            improved = True
                            break
    return path.flatten(), best_area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # Safety margin (60s buffer)
    
    # 1. Generate initial heuristic path - use a smoother, wider arc
    n_ctrl = 12
    t = np.linspace(0, 1, n_ctrl)
    # Use a raised cosine for smoother motion
    x = -2.0 * t
    y = 1.5 * np.sin(np.pi * t)  # Wider arc to allow rotation
    angles = 90.0 * t
    
    initial_params = np.column_stack([x, y, angles]).flatten()
    
    # 2. Global Optimization on very coarse Subgrid (step=5)
    # ~1/25 the points, much faster
    sub_x, sub_y = _build_grid(step=5)
    
    best_params = initial_params.copy()
    best_area_sub = _evaluate_path_fast(best_params.reshape(-1, 3), sub_x, sub_y)
    
    # Nelder-Mead on coarse subgrid - many iterations
    try:
        opt_params, opt_area = _nelder_mead_optimize(
            initial_params, sub_x, sub_y, 
            deadline=deadline, 
            initial_step=0.8, 
            max_iter=500,
            eval_func=_evaluate_path_fast
        )
        if opt_area > best_area_sub:
            best_params = opt_params
            best_area_sub = opt_area
    except Exception as e:
        pass
        
    # 3. Intermediate Optimization on Subgrid (step=2)
    mid_x, mid_y = _build_grid(step=2)
    
    best_area_mid = _evaluate_path_fast(best_params.reshape(-1, 3), mid_x, mid_y)
    
    # Coordinate Descent on mid grid
    try:
        cd_params, cd_area = _coordinate_descent(
            best_params, mid_x, mid_y, 
            deadline=deadline,
            step_sizes=[0.3, 0.15, 0.08, 0.04],
            eval_func=_evaluate_path_fast
        )
        if cd_area > best_area_mid:
            best_params = cd_params
            best_area_mid = cd_area
    except Exception as e:
        pass
        
    # 4. Final Refinement on Full Grid (step=1)
    full_x, full_y = _build_grid(step=1)
    
    best_area_full = _evaluate_path_fast(best_params.reshape(-1, 3), full_x, full_y)
    
    # Coordinate Descent on full grid - fine steps
    try:
        final_params, final_area = _coordinate_descent(
            best_params, full_x, full_y, 
            deadline=deadline,
            step_sizes=[0.05, 0.02, 0.01, 0.005, 0.002],
            eval_func=_evaluate_path_fast
        )
        if final_area > best_area_full:
            best_params = final_params
            best_area_full = final_area
    except Exception as e:
        pass
        
    # 5. High-precision polish: very fine coordinate descent
    if time.time() < deadline - 30:
        try:
            polish_params, polish_area = _coordinate_descent(
                best_params, full_x, full_y, 
                deadline=deadline,
                step_sizes=[0.001, 0.0005, 0.0002],
                eval_func=_evaluate_path_fast
            )
            if polish_area > best_area_full:
                best_params = polish_params
                best_area_full = polish_area
        except Exception as e:
            pass
    
    # 6. Try alternative initial paths if time remains
    if time.time() < deadline - 60:
        # Alternative: more control points, different arc
        n_ctrl2 = 16
        t2 = np.linspace(0, 1, n_ctrl2)
        x2 = -2.0 * t2
        y2 = 1.8 * np.sin(np.pi * t2)
        angles2 = 90.0 * t2
        init2 = np.column_stack([x2, y2, angles2]).flatten()
        
        try:
            opt2, area2_sub = _nelder_mead_optimize(
                init2, sub_x, sub_y, 
                deadline=deadline, 
                initial_step=0.5, 
                max_iter=200,
                eval_func=_evaluate_path_fast
            )
            area2_full = _evaluate_path_fast(opt2.reshape(-1, 3), full_x, full_y)
            if area2_full > best_area_full:
                best_params = opt2
                best_area_full = area2_full
                # Refine
                final2, area2_ref = _coordinate_descent(
                    best_params, full_x, full_y, 
                    deadline=deadline,
                    step_sizes=[0.05, 0.02, 0.01],
                    eval_func=_evaluate_path_fast
                )
                if area2_ref > best_area_full:
                    best_params = final2
                    best_area_full = area2_ref
        except Exception as e:
            pass
            
    return best_params.reshape(-1, 3)

# EVOLVE-BLOCK-END
