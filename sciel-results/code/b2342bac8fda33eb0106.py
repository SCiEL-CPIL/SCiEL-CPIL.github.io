import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 50s margin before 630s kill

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    # B-spline parameterization: control points define a smooth curve
    # Parameters: for each control point, (x, y) - angle is determined by parameter t
    # We use 8 control points, with t values fixed at evenly spaced positions
    n_ctrl = 8
    t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    
    # Initial control points (x, y) - will be optimized
    # Start from a reasonable path
    init_xy = np.array([
        [0.0, 0.0],
        [-0.1, 0.1],
        [-0.25, 0.25],
        [-0.4, 0.3],
        [-0.55, 0.25],
        [-0.7, 0.15],
        [-0.85, 0.05],
        [-1.0, 0.0],
    ])
    
    def params_to_path(params):
        """Convert control point parameters to a dense path."""
        # params = [x0, y0, x1, y1, ..., x7, y7]
        xy_ctrl = params.reshape(n_ctrl, 2)
        
        # Create cubic spline for x(t) and y(t)
        cs_x = CubicSpline(t_ctrl, xy_ctrl[:, 0])
        cs_y = CubicSpline(t_ctrl, xy_ctrl[:, 1])
        
        # Sample the spline at many points
        n_samples = 200
        t_samples = np.linspace(0.0, 1.0, n_samples)
        x_samples = cs_x(t_samples)
        y_samples = cs_y(t_samples)
        
        # Angle is linear in t: a(t) = t * 90
        a_samples = t_samples * 90.0
        
        # Build path: (x, y, a) for each sample
        path = np.column_stack([x_samples, y_samples, a_samples])
        
        # Ensure endpoints match exactly
        path[0] = [0.0, 0.0, 0.0]
        path[-1] = [-2.0, 0.0, 90.0]
        
        return path

    def densify_path(path):
        """Add linear interpolation between consecutive placements."""
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

    def count_inside(placements):
        """Count grid points inside all placements using progressive pruning."""
        valid_indices = np.arange(n_points)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            Xv = X_flat[valid_indices]
            Yv = Y_flat[valid_indices]
            
            dx = Xv - x
            dy = Yv - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
            in_rect2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            valid_indices = valid_indices[inside]
            if len(valid_indices) == 0:
                return 0
        
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = count_inside(dense)
        return count * 0.0001

    # Initial path from B-spline
    init_params = init_xy.flatten()
    best_path = params_to_path(init_params)
    best_score = score_path(best_path)
    print(f"Initial B-spline score: {best_score:.6f}")

    def objective(params):
        """Minimize negative score. Uses smooth soft-min surrogate for gradient info."""
        if time.time() > deadline:
            return 0.0
        
        path = params_to_path(params)
        
        # For SLSQP, we need a smooth objective. 
        # The count is integer-valued, so we use a smooth approximation.
        # We'll compute the "margin" for each grid point across all placements,
        # and use a soft-min to approximate the minimum margin (which determines if a point is inside all).
        
        dense = densify_path(path)
        
        # Compute margins for all points in all placements
        # margin > 0 means inside, margin <= 0 means outside
        # We want to maximize the number of points with min_margin > 0
        # Approximate with: sum of softplus(min_margin) / scale
        
        # To keep it fast, we'll use a subsample of points
        if n_points > 5000:
            indices = np.random.choice(n_points, 5000, replace=False)
            Xs = X_flat[indices]
            Ys = Y_flat[indices]
            n_sub = len(indices)
        else:
            Xs = X_flat
            Ys = Y_flat
            n_sub = n_points
        
        min_margins = np.full(n_sub, np.inf)
        
        for x, y, a_deg in dense:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = Xs - x
            dy = Ys - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            # Distance to complement of base corridor (negative margin = outside)
            # Base is union of rect1 and rect2
            # For rect1: 0<=u<=1, -3<=v<=1
            # For rect2: -3<=u<=1, 0<=v<=1
            # A point is outside if it's outside both rects
            # margin for rect1 = min(u, 1-u, v+3, 1-v)
            # margin for rect2 = min(u+3, 1-u, v, 1-v)
            # point is inside if max(margin1, margin2) > 0
            
            margin1 = np.minimum.reduce([u, 1 - u, v + 3, 1 - v])
            margin2 = np.minimum.reduce([u + 3, 1 - u, v, 1 - v])
            margin = np.maximum(margin1, margin2)
            
            min_margins = np.minimum(min_margins, margin)
        
        # Use soft approximation: count of points with min_margin > threshold
        # Soft count: sum of sigmoid(k * min_margin)
        k = 50.0  # sharpness
        soft_count = np.sum(1.0 / (1.0 + np.exp(-k * min_margins)))
        
        # Scale to approximate area
        if n_sub < n_points:
            estimated_count = soft_count * (n_points / n_sub)
        else:
            estimated_count = soft_count
        
        return -estimated_count * 0.0001

    # Bounds for control points
    # x should be in roughly [-1.5, 0.5], y in [-0.5, 1.0]
    bounds = [(-1.5, 0.5)] * n_ctrl + [(-0.5, 1.0)] * n_ctrl

    # Optimize control points using SLSQP
    best_params = init_params.copy()
    
    # Timing a single evaluation
    t0 = time.time()
    objective(init_params)
    t1 = time.time()
    avg_eval_time = t1 - t0
    print(f"Estimated avg eval time: {avg_eval_time*1000:.1f}ms")
    
    time_left = deadline - time.time()
    max_iter = max(20, int(time_left * 0.7 / avg_eval_time))
    
    # SLSQP optimization
    try:
        result = minimize(
            objective,
            init_params,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.fun < 0:
            candidate_score = -result.fun
            # Validate with exact count
            candidate_path = params_to_path(result.x)
            exact_score = score_path(candidate_path)
            print(f"SLSQP result: smooth={candidate_score:.6f}, exact={exact_score:.6f}")
            
            if exact_score > best_score:
                best_params = result.x
                best_path = candidate_path
                best_score = exact_score
        else:
            print(f"SLSQP did not improve: {result.fun:.6f}")
    except Exception as e:
        print(f"SLSQP failed: {e}")
    
    # Local refinement with Nelder-Mead on the exact objective
    if time.time() < deadline - 60:
        try:
            def exact_objective(params):
                if time.time() > deadline:
                    return 0.0
                path = params_to_path(params)
                return -score_path(path)
            
            nm_result = minimize(
                exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            nm_score = -nm_result.fun
            if nm_score > best_score:
                best_params = nm_result.x
                best_path = params_to_path(best_params)
                best_score = nm_score
            print(f"Nelder-Mead result: {best_score:.6f}")
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
    
    # Multi-start with perturbation
    if time.time() < deadline - 30:
        for trial in range(10):
            if time.time() > deadline - 15:
                break
            
            # Perturb the best solution
            perturbation = np.random.randn(len(best_params)) * 0.05
            candidate = np.clip(best_params + perturbation, 
                                [b[0] for b in bounds], 
                                [b[1] for b in bounds])
            
            # Quick local optimization from this candidate
            try:
                local_result = minimize(
                    exact_objective,
                    candidate,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                local_score = -local_result.fun
                if local_score > best_score:
                    best_params = local_result.x
                    best_path = params_to_path(best_params)
                    best_score = local_score
                    print(f"  Trial {trial}: improved to {best_score:.6f}")
            except Exception:
                pass
    
    # Final coordinate ascent polish
    if time.time() < deadline - 10:
        current = best_params.copy()
        current_score = best_score
        step = 0.05
        improved = True
        while improved and step > 1e-5 and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                for direction in [-1, 1]:
                    new_val = current[i] + direction * step
                    if new_val < bounds[i][0] or new_val > bounds[i][1]:
                        continue
                    old_val = current[i]
                    current[i] = new_val
                    new_score = -exact_objective(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = old_val
            step *= 0.5
            if improved and time.time() < deadline - 5:
                if current_score > best_score:
                    best_params = current.copy()
                    best_path = params_to_path(current)
                    best_score = current_score
    
    # Final validation
    final_score = score_path(best_path)
    if final_score > best_score:
        best_score = final_score
    
    print(f"Final score: {best_score:.6f}")
    
    return best_path
