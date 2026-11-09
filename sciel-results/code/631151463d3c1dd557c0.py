import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # We will optimize a dense path of waypoints.
    # The evaluator inserts (0,0,0) at start and (-2,0,90) at end.
    # It interpolates between consecutive rows in `m = max(1, ceil(|a2 - a1|))` steps.
    # If we provide rows at angles 2, 4, 6, ..., 88 (45 points), then between 0 and 2, m=2.
    # Between 2 and 4, m=2.
    # This creates a dense path of 1 + 45*2 + 1 = 92 placements.
    # Actually, let's just provide every 2 degrees.
    # Angles: 2, 4, ..., 88.
    # Number of points: 45.
    # Parameters: x[0..44], y[0..44]. Total 90 params.
    
    angles = np.arange(2, 90, 2)
    n_waypoints = len(angles)
    
    # Initial guess: Linear interpolation from (0,0) to (-2,0) with a bulge in y.
    # Let's start with a straight line and add a sinusoidal bulge.
    # x(t) = -2*t, y(t) = H * sin(pi*t) where t in [0,1].
    # t = angle / 90.
    t_init = angles / 90.0
    x_init = -2.0 * t_init
    y_init = 1.0 * np.sin(np.pi * t_init) # Amplitude 1.0
    
    params = np.concatenate([x_init, y_init])
    
    def evaluate_params(params, active_mask=None):
        if active_mask is None:
            active_mask = np.ones(n_points, dtype=bool)
        else:
            active_mask = active_mask.copy()
            
        x_vals = params[:n_waypoints]
        y_vals = params[n_waypoints:]
        
        # Construct full path including endpoints for interpolation logic
        # The evaluator logic:
        # 1. Prepend (0,0,0)
        # 2. Append (-2,0,90)
        # 3. Interpolate between consecutive.
        
        # To match evaluator exactly, we build the dense path.
        # But for speed, we can just iterate through the segments.
        
        # Segment 0: (0,0,0) to (x[0], y[0], 2)
        # m = ceil(2) = 2.
        # Interpolations: k=1,2.
        # k=1: t=0.5 -> (x[0]/2, y[0]/2, 1)
        # k=2: t=1.0 -> (x[0], y[0], 2)
        
        # Segment i: (x[i-1], y[i-1], 2i) to (x[i], y[i], 2i+2)
        # m = 2.
        # k=1: t=0.5 -> midpoint, angle 2i+1
        # k=2: t=1.0 -> (x[i], y[i], 2i+2)
        
        # Segment last: (x[-1], y[-1], 88) to (-2, 0, 90)
        # m = 2.
        # k=1: t=0.5 -> midpoint, angle 89
        # k=2: t=1.0 -> (-2, 0, 90)
        
        # Total placements:
        # Start (0,0,0)
        # For each of 45 waypoints:
        #   Midpoint (angle odd)
        #   Endpoint (angle even)
        # End is included in last endpoint.
        
        # Let's generate the dense list of (x, y, a)
        dense_path = [(0.0, 0.0, 0.0)]
        
        for i in range(n_waypoints):
            x1 = x_vals[i]
            y1 = y_vals[i]
            a1 = angles[i]
            
            if i == 0:
                px_prev, py_prev, pa_prev = 0.0, 0.0, 0.0
            else:
                px_prev, py_prev, pa_prev = x_vals[i-1], y_vals[i-1], angles[i-1]
                
            # Interpolate from prev to current
            # m = 2
            # k=1: 0.5
            mx = (px_prev + x1) / 2.0
            my = (py_prev + y1) / 2.0
            ma = (pa_prev + a1) / 2.0
            dense_path.append((mx, my, ma))
            
            # k=2: 1.0 (Current point)
            dense_path.append((x1, y1, a1))
            
        # Final segment to (-2, 0, 90)
        px_prev, py_prev, pa_prev = x_vals[-1], y_vals[-1], angles[-1]
        ex, ey, ea = -2.0, 0.0, 90.0
        
        mx = (px_prev + ex) / 2.0
        my = (py_prev + ey) / 2.0
        ma = (pa_prev + ea) / 2.0
        dense_path.append((mx, my, ma))
        dense_path.append((ex, ey, ea))
        
        # Evaluate intersection
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
            
            active_mask &= in_base
            
            # Early termination if intersection becomes empty or too small
            # The sofa area is ~1.0 (area of L-shape). Grid points ~10000.
            # If count drops below 500, it's likely failing.
            if np.sum(active_mask) < 500:
                return 0.0, active_mask
                
        score = np.sum(active_mask) * 0.0001
        return score, active_mask

    # Coarse-to-fine Coordinate Descent
    # We have 90 parameters.
    # Step 1: Coarse grid. Step size 0.1.
    # Step 2: Fine grid. Step size 0.01.
    # Step 3: Very fine. Step size 0.001.
    
    best_score, best_mask = evaluate_params(params)
    best_params = params.copy()
    
    step_sizes = [0.2, 0.1, 0.05, 0.01, 0.002]
    
    for step in step_sizes:
        if time.time() > deadline - 10:
            break
            
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(len(best_params)):
                orig_val = best_params[i]
                best_local_val = orig_val
                best_local_score = best_score
                best_local_mask = best_mask
                
                # Try +/- step
                for delta in [-step, step]:
                    new_val = orig_val + delta
                    # Optional: bounds. 
                    # x should be in [-3, 1], y in [-1.5, 1.5] roughly.
                    # Let's not bound strictly to allow exploration, but clamp to reasonable range to avoid NaNs or huge values.
                    if new_val < -5 or new_val > 5:
                        continue
                        
                    test_params = best_params.copy()
                    test_params[i] = new_val
                    score, mask = evaluate_params(test_params, best_mask)
                    
                    if score > best_local_score:
                        best_local_score = score
                        best_local_val = new_val
                        best_local_mask = mask
                        improved = True
                        
                if best_local_val != orig_val:
                    best_params[i] = best_local_val
                    best_score = best_local_score
                    best_mask = best_local_mask
                    
    # Final check
    final_score, _ = evaluate_params(best_params)
    if final_score > best_score:
        best_score = final_score

    # Construct final path
    x_vals = best_params[:n_waypoints]
    y_vals = best_params[n_waypoints:]
    
    final_path = np.column_stack([x_vals, y_vals, angles])
    
    return final_path
