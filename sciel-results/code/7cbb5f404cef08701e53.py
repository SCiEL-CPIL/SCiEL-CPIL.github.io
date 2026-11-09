import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # Pre-compute dense path angles for the evaluator's logic
    # The evaluator adds (0,0,0) and (-2,0,90).
    # Our path defines the intermediate points.
    # To make optimization efficient, we use a parametric curve.
    # Let's use a polynomial or trigonometric series for x(t) and y(t).
    # t in [0, 1] corresponding to angle 0 to 90.
    
    # Parameters: coefficients for x(t) and y(t)
    # x(t) = c_x0 + c_x1*t + c_x2*t^2 + c_x3*t^3
    # y(t) = c_y0 + c_y1*t + c_y2*t^2 + c_y3*t^3
    # We fix c_x0=0, c_y0=0, c_x3=-2 (to ensure end point is -2? No, endpoint is fixed by evaluator, but our last point should be close to it or exactly it to avoid interpolation issues? 
    # Actually, the evaluator interpolates between our last point and (-2,0,90). 
    # If we make our last point exactly (-2,0,90), the interpolation step is trivial.
    # However, the angle a in our path is also a variable? 
    # The current program fixes angles to 1..89. 
    # Let's stick to fixed angles 1..89 to reduce dimensionality, but optimize x,y.
    # Or better: optimize the control points of a Bezier curve.
    
    # Let's use 4 control points for x and 4 for y (Cubic Bezier).
    # P0 = (0,0), P3 = (-2,0).
    # P1 = (x1, y1), P2 = (x2, y2).
    # This gives 4 parameters. Very low dimensionality.
    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    # We sample this curve at angles 1..89.
    # t = angle / 90.
    
    def evaluate_params(params):
        x1, y1, x2, y2 = params
        
        # Bezier control points
        P0 = np.array([0.0, 0.0])
        P1 = np.array([x1, y1])
        P2 = np.array([x2, y2])
        P3 = np.array([-2.0, 0.0])
        
        # Sample at angles 1 to 89
        angles = np.arange(1, 90)
        t = angles / 90.0
        
        # Compute points on the Bezier curve
        # x(t) = (1-t)^3*0 + 3(1-t)^2*t*x1 + 3(1-t)*t^2*x2 + t^3*(-2)
        # y(t) = (1-t)^3*0 + 3(1-t)^2*t*y1 + 3(1-t)*t^2*y2 + t^3*0
        
        # Vectorized computation
        t_arr = t
        one_minus_t = 1 - t_arr
        
        x_vals = 3 * (one_minus_t**2) * t_arr * x1 + 3 * (one_minus_t) * (t_arr**2) * x2 - 2 * (t_arr**3)
        y_vals = 3 * (one_minus_t**2) * t_arr * y1 + 3 * (one_minus_t) * (t_arr**2) * y2
        
        path = np.column_stack([x_vals, y_vals, angles])
        
        # Full path including endpoints
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t_interp = k / m
                    interp = p1 + t_interp * (p2 - p1)
                    dense_path.append(interp)
                    
        dense_path = np.array(dense_path)
        
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            count = np.sum(valid)
            if count < 100:
                return 0.0
                
        score = np.sum(valid) * 0.0001
        return score

    def negative_score(params):
        return -evaluate_params(params)

    # Multi-start optimization
    # Initial guesses based on known sofa moving heuristics
    # The sofa moves in an S-curve.
    # P1 should be in the first quadrant relative to start?
    # Start (0,0), End (-2,0).
    # The "bulge" should be towards +y.
    
    initial_guesses = [
        [0.5, 1.0, -1.5, 1.0],
        [0.0, 1.2, -2.0, 1.2],
        [0.5, 0.8, -1.5, 0.8],
        [1.0, 1.5, -2.5, 1.5],
        [0.0, 0.0, -2.0, 0.0], # Straight line (bad)
        [0.5, 1.5, -1.5, 0.5], # Asymmetric
        [0.0, 1.0, -2.0, 1.0],
        [0.5, 1.0, -2.0, 1.0],
    ]
    
    best_score = 0.0
    best_params = initial_guesses[0]
    
    # Bounds for control points
    # x1, x2 should be between -3 and 1
    # y1, y2 should be between -1.5 and 1.5 (sofa height is 1.5 roughly, but can go higher? 
    # The corridor is width 1. The sofa is 1x3 L-shape.
    # The max height of the sofa in the corridor is 1.
    # So y should probably stay within [-1, 1] or slightly more to allow rotation?
    # Let's be generous: [-2, 2]
    bounds = [(-3.0, 1.0), (-2.0, 2.0), (-3.0, 1.0), (-2.0, 2.0)]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(negative_score, guess, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-6, 'maxfun': 1000})
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception as e:
            pass

    # If L-BFGS-B didn't find anything better, try a few random local searches from the best
    if time.time() < deadline - 60:
        current_params = best_params.copy()
        current_score = best_score
        
        # Simple coordinate descent refinement
        step_size = 0.05
        for _ in range(20): # Max 20 passes
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(4):
                orig_val = current_params[i]
                best_local_val = orig_val
                best_local_score = current_score
                
                for delta in [-step_size, step_size]:
                    new_val = orig_val + delta
                    # Clamp
                    new_val = np.clip(new_val, bounds[i][0], bounds[i][1])
                    
                    test_params = current_params.copy()
                    test_params[i] = new_val
                    score = evaluate_params(test_params)
                    
                    if score > best_local_score:
                        best_local_score = score
                        best_local_val = new_val
                        improved = True
                        
                current_params[i] = best_local_val
                if best_local_score > current_score:
                    current_score = best_local_score
                    
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()

    # Construct final path from best_params
    x1, y1, x2, y2 = best_params
    
    angles = np.arange(1, 90)
    t = angles / 90.0
    one_minus_t = 1 - t
    
    x_vals = 3 * (one_minus_t**2) * t * x1 + 3 * (one_minus_t) * (t**2) * x2 - 2 * (t**3)
    y_vals = 3 * (one_minus_t**2) * t * y1 + 3 * (one_minus_t) * (t**2) * y2
    
    final_path = np.column_stack([x_vals, y_vals, angles])
    
    return final_path
