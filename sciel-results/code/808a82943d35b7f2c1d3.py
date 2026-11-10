import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 50s margin for safety

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    # Precompute angles for the dense path
    n_steps = 400
    angles = np.linspace(0.0, 90.0, n_steps)
    cos_a = np.cos(np.deg2rad(angles))
    sin_a = np.sin(np.deg2rad(angles))

    def densify_path(path):
        # path is (n_steps, 3)
        full_path = np.vstack([P_start, path, P_end])
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        return np.array(dense_points)

    def check_inside_fast(placements):
        # Adaptive mask
        valid_mask = np.ones(n_points, dtype=bool)
        valid_indices = np.arange(n_points)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Only check valid points
            Xv = X_flat[valid_indices]
            Yv = Y_flat[valid_indices]
            
            dx = Xv - x
            dy = Yv - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            # Update valid indices
            valid_indices = valid_indices[inside]
            
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = check_inside_fast(dense)
        return count * 0.0001

    # Initial best path (from seed)
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    best_score = score_path(best_path)

    # Define optimization function
    # We will optimize the x,y positions at n_steps fixed angles: 0, 90/n_steps, ..., 90
    # Angles 0 and 90 are fixed by the evaluator, so we optimize n_steps intermediate points
    # That's n_steps points * 2 = 2*n_steps variables.
    # But we can fix the first and last points to match the evaluator's start/end if we want, 
    # but the evaluator adds P_start and P_end. Our path is inserted between.
    # The evaluator does: P_start, [our path], P_end.
    # So our path should ideally start near P_start and end near P_end.
    # However, the angle in our path must be monotonic from 0 to 90? 
    # The evaluator interpolates linearly between consecutive rows.
    # If we provide a path with angles 0, 90/n, ..., 90, the interpolation steps will be small.
    # The evaluator adds P_start (0,0,0) before. If our first point is (x0, y0, 0), 
    # the step from (0,0,0) to (x0, y0, 0) is m=1 (delta a = 0). 
    # Wait, m = max(1, ceil(|a2-a1|)). If a2=a1=0, m=1. One interpolation step.
    # This is fine.
    
    # Let's optimize x and y for n_steps points.
    # The angles are fixed to linspace(0, 90, n_steps).
    
    def path_from_params(params):
        # params: [x0, y0, x1, y1, ..., xn, yn]
        path = np.zeros((n_steps, 3))
        path[:, 0] = params[0::2]
        path[:, 1] = params[1::2]
        path[:, 2] = angles
        return path

    def objective(params):
        if time.time() > deadline:
            return 1e9 # Large penalty
        path = path_from_params(params)
        return -score_path(path) # Minimize negative score

    # Bounds for x, y
    # The sofa must stay inside the corridor. 
    # The corridor is roughly in the range [-3, 1] in u and v.
    # The inner corner is at (x,y).
    # For a=0, the corridor is to the right and above (x,y).
    # For a=90, the corridor is to the left and above (x,y) (rotated).
    # x should probably be between -2 and 1.
    # y should probably be between -1 and 2.
    bounds = [(-2.0, 1.0), (-1.0, 2.0)] * n_steps

    # Initial guess
    # Interpolate the seed path to n_steps points
    angles_seed = best_path[:, 2]
    x_interp = np.interp(angles, angles_seed, best_path[:, 0])
    y_interp = np.interp(angles, angles_seed, best_path[:, 1])
    initial_params = np.column_stack((x_interp, y_interp)).flatten()

    # Nelder-Mead optimization
    # This is high dimensional (800 variables). Nelder-Mead might be slow or get stuck.
    # But the problem is smooth.
    # Let's try a coarser grid first? 
    # The task requires returning a float array. 
    # The evaluator densifies based on angle difference.
    # If we use n_steps=100, the angle step is 0.9 degrees. m=1.
    # If we use n_steps=400, the angle step is 0.225 degrees. m=1.
    # The cost of evaluation is O(n_points * n_placements).
    # n_points ~ 135000. n_placements ~ 400.
    # 135000 * 400 = 54,000,000 operations.
    # In Python/NumPy, this is fast (vectorized).
    # Let's measure the time for one evaluation.
    
    t0 = time.time()
    score_path(path_from_params(initial_params))
    t1 = time.time()
    eval_time = t1 - t0
    print(f"Eval time: {eval_time:.4f} s")
    
    # Nelder-Mead with 800 variables is likely too slow or unstable.
    # Let's reduce n_steps for optimization, then refine?
    # Or use a different optimizer?
    # The prompt suggests: "Represent the path as a smooth parametric curve defined by a small set of control points using cubic B-splines, and optimize these control points directly with SLSQP".
    
    # Let's implement the B-spline strategy.
    # Control points: K points.
    # We optimize K x (x, y) coordinates.
    # The path is generated by evaluating the B-spline at n_eval points (e.g., 200).
    # Then we densify that path.
    
    K = 10 # Number of control points
    # B-spline knots: uniform
    # We want the curve to start at (0,0) and end at (-2,0) roughly?
    # Actually, the evaluator adds P_start and P_end.
    # So our path just needs to be a good continuation.
    # But the angle must go from 0 to 90.
    # Let's parameterize the B-spline by t in [0, 1].
    # x(t), y(t).
    # Angle is fixed by t: a(t) = 90 * t.
    # So we define x(t) and y(t) using B-splines.
    
    # Control points: P0, P1, ..., P_{K-1}
    # We want x(0) to be close to 0 and x(1) close to -2?
    # Not necessarily, the evaluator handles the start and end.
    # But it's good to be smooth.
    
    # Let's use scipy.interpolate.BSpline or just simple cubic interpolation?
    # Cubic B-spline with clamped knots.
    
    # Simpler: Use a set of waypoints and interpolate with cubic spline?
    # The prompt specifically says "cubic B-splines".
    
    # Let's define the B-spline basis functions or use scipy.
    # scipy.interpolate.BSpline(t, c, k)
    # t: knots, c: coefficients (control points), k: order (degree+1)
    
    # We want to map t in [0, 1] to x and y.
    # Let's use K=10 control points.
    # Knots: [0, 0, 0, 0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1, 1, 1, 1] ?
    # Degree 3, so 4 knots at start and end.
    # Number of knots = K + degree + 1 = 10 + 3 + 1 = 14.
    # Knots: 0, 0, 0, 0, 1/9, 2/9, ..., 8/9, 1, 1, 1, 1
    
    knots = np.array([0, 0, 0, 0] + list(np.linspace(1/(K-1), 1-1/(K-1), K-2)) + [1, 1, 1, 1])
    
    # Coefficients are the control points.
    # We optimize 2*K parameters: x0..xK-1, y0..yK-1
    
    def bspline_path_from_params(params):
        x_ctrl = params[:K]
        y_ctrl = params[K:]
        
        # Create splines
        # Note: BSpline expects coefficients in the same order as knots.
        # For clamped splines, the first and last 'degree' coefficients are the endpoints?
        # No, for clamped splines, the curve passes through the first and last control points.
        
        try:
            from scipy.interpolate import BSpline
            sx = BSpline(knots, x_ctrl, 3)
            sy = BSpline(knots, y_ctrl, 3)
            
            # Evaluate at n_eval points
            t_eval = np.linspace(0, 1, n_steps)
            x_eval = sx(t_eval)
            y_eval = sy(t_eval)
            
            path = np.zeros((n_steps, 3))
            path[:, 0] = x_eval
            path[:, 1] = y_eval
            path[:, 2] = angles
            return path
        except Exception:
            # Fallback to linear interpolation if spline fails
            x_interp = np.interp(t_eval, np.linspace(0, 1, K), x_ctrl)
            y_interp = np.interp(t_eval, np.linspace(0, 1, K), y_ctrl)
            path = np.zeros((n_steps, 3))
            path[:, 0] = x_interp
            path[:, 1] = y_interp
            path[:, 2] = angles
            return path

    def objective_bspline(params):
        if time.time() > deadline:
            return 1e9
        path = bspline_path_from_params(params)
        return -score_path(path)

    # Initial guess for control points
    # Interpolate the seed path to K points
    t_ctrl = np.linspace(0, 1, K)
    a_ctrl = 90 * t_ctrl
    x_ctrl_init = np.interp(a_ctrl, angles_seed, best_path[:, 0])
    y_ctrl_init = np.interp(a_ctrl, angles_seed, best_path[:, 1])
    initial_params_bspline = np.concatenate([x_ctrl_init, y_ctrl_init])

    # SLSQP optimization
    # SLSQP can handle 20 variables easily.
    
    # Bounds
    bounds_bspline = [(-2.0, 1.0)] * K + [(-1.0, 2.0)] * K

    print("Starting SLSQP optimization")
    
    best_params_bspline = None
    try:
        result = minimize(
            objective_bspline,
            initial_params_bspline,
            method='SLSQP',
            bounds=bounds_bspline,
            options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
        )
        
        if result.fun < 1e8:
            best_params_bspline = result.x
            best_path_bspline = bspline_path_from_params(best_params_bspline)
            best_score_bspline = -result.fun
            if best_score_bspline > best_score:
                best_path = best_path_bspline
                best_score = best_score_bspline
                print(f"SLSQP improved score to {best_score}")
    except Exception as e:
        print(f"SLSQP failed: {e}")

    # Local refinement using Nelder-Mead on the control points
    if time.time() < deadline - 30:
        print("Starting local refinement with Nelder-Mead")
        
        if best_params_bspline is not None:
            current_params = best_params_bspline
        else:
            current_params = initial_params_bspline
            
        current_score = -objective_bspline(current_params)
        
        try:
            nm_result = minimize(
                objective_bspline,
                current_params,
                method='Nelder-Mead',
                bounds=bounds_bspline,
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if nm_result.fun < current_score:
                current_params = nm_result.x
                current_score = nm_result.fun
                print(f"Nelder-Mead improved score to {-current_score}")
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
        
        final_path = bspline_path_from_params(current_params)
        final_score = score_path(final_path)
        if final_score > best_score:
            best_path = final_path
            best_score = final_score

    return best_path
