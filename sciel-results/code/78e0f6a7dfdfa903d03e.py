import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """Densify a path (n,3) array using linear interpolation between consecutive points."""
        if len(path) <= 1:
            return np.vstack([P_start, path, P_end])
        full_path = np.vstack([P_start, path, P_end])
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        return np.array(dense_points)

    def check_inside_fast(placements):
        """Count sofa points inside all placements using adaptive mask."""
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

            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2

            if not np.any(inside):
                return 0

            valid_indices = valid_indices[inside]

        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = check_inside_fast(dense)
        return count * 0.0001

    # B-spline parameterization:
    # We define a cubic B-spline for x(a) and y(a) over angle a in [0, 90].
    # Control points: we use 5 internal knots at angles [15, 30, 45, 60, 75].
    # The spline must pass through (0,0) at a=0 and (-2,0) at a=90.
    # We optimize the x,y values at the 5 internal knots.
    # Total variables: 10 (5 x-values + 5 y-values).

    n_knots = 5
    knot_angles = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
    
    # Full knot sequence including endpoints
    all_angles = np.concatenate([[0.0], knot_angles, [90.0]])
    all_x = np.concatenate([[0.0], np.zeros(n_knots), [-2.0]])  # placeholder, will be set
    all_y = np.concatenate([[0.0], np.zeros(n_knots), [0.0]])

    def path_from_spline_params(params):
        """Convert spline control point params to a dense path."""
        # params: [x1, y1, x2, y2, ..., x5, y5]
        x_vals = np.concatenate([[0.0], params[0::2], [-2.0]])
        y_vals = np.concatenate([[0.0], params[1::2], [0.0]])
        
        # Create cubic spline
        try:
            cs_x = CubicSpline(all_angles, x_vals, bc_type='not-a-knot')
            cs_y = CubicSpline(all_angles, y_vals, bc_type='not-a-knot')
        except Exception:
            # Fallback: linear interpolation
            x_vals = np.interp(all_angles, all_angles, x_vals)
            y_vals = np.interp(all_angles, all_angles, y_vals)
            return np.column_stack([x_vals, y_vals, all_angles])
        
        # Sample the spline at fine resolution
        sample_angles = np.arange(0.0, 90.01, 1.0)
        x_samples = cs_x(sample_angles)
        y_samples = cs_y(sample_angles)
        
        path = np.column_stack([x_samples, y_samples, sample_angles])
        return path

    def objective(params):
        if time.time() > deadline:
            return 0.0
        path = path_from_spline_params(params)
        return -score_path(path)

    # Initial guess: smooth curve from (0,0) to (-2,0)
    # x goes from 0 to -2, y stays near 0 with slight bulge
    init_x = np.array([-0.2, -0.5, -0.8, -1.1, -1.4])
    init_y = np.array([0.1, 0.15, 0.15, 0.1, 0.05])
    init_params = np.concatenate([init_x, init_y])

    # Bounds: x in [-2.5, 0.5], y in [-1.0, 1.0]
    bounds = [(-2.5, 0.5), (-1.0, 1.0)] * n_knots

    # SLSQP optimization
    print("Starting SLSQP optimization")
    t0 = time.time()
    try:
        slsqp_result = minimize(
            objective,
            init_params,
            method='SLSQP',
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-6,
                'disp': False
            }
        )
        print(f"SLSQP done in {time.time()-t0:.1f}s, fun={slsqp_result.fun:.6f}")
        if slsqp_result.fun < 0:
            best_params = slsqp_result.x
            best_score = -slsqp_result.fun
        else:
            best_params = init_params
            best_score = score_path(path_from_spline_params(init_params))
    except Exception as e:
        print(f"SLSQP failed: {e}")
        best_params = init_params
        best_score = score_path(path_from_spline_params(init_params))

    # Nelder-Mead polish
    if time.time() < deadline - 60:
        print("Starting Nelder-Mead polish")
        try:
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'maxfev': 1000,
                    'xatol': 0.001,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            print(f"NM done, fun={nm_result.fun:.6f}")
            if nm_result.fun < 0 and -nm_result.fun > best_score:
                best_params = nm_result.x
                best_score = -nm_result.fun
        except Exception as e:
            print(f"NM failed: {e}")

    # Final high-precision polish with smaller steps
    if time.time() < deadline - 30:
        print("Starting fine-tuning")
        current_params = best_params.copy()
        current_score = -objective(current_params)
        
        rng = np.random.default_rng(42)
        step_sizes = [0.02, 0.01, 0.005, 0.002]
        
        for step_size in step_sizes:
            if time.time() > deadline - 10:
                break
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for _ in range(30):
                    if time.time() > deadline - 10:
                        break
                    perturbation = rng.normal(0, step_size, size=current_params.shape)
                    new_params = current_params + perturbation
                    new_params[0::2] = np.clip(new_params[0::2], -2.5, 0.5)
                    new_params[1::2] = np.clip(new_params[1::2], -1.0, 1.0)
                    
                    new_score = -objective(new_params)
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_params = current_params
                            best_score = current_score

    final_path = path_from_spline_params(best_params)
    final_score = score_path(final_path)
    if final_score > best_score:
        best_path = final_path
        best_score = final_score
    else:
        best_path = final_path

    print(f"Final score: {best_score:.6f}")
    return best_path
