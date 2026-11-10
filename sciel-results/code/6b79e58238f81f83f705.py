import numpy as np
import time
import scipy.optimize as opt

def moving_sofa_2d() -> np.ndarray:
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)
    
    start_time = time.time()
    time_limit = 580  # Safety margin before 630s kill

    def evaluate_path(path):
        # path: list of (x, y, a)
        # Simulate evaluator interpolation
        full_path = [(0.0, 0.0, 0.0)]
        for p in path:
            full_path.append(p)
        full_path.append((-2.0, 0.0, 90.0))
        
        interp_points = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                a = p1[2] + (p2[2] - p1[2]) * t
                interp_points.append((x, y, a))
        interp_points.append(full_path[-1])
        
        mask = np.ones(n_points, dtype=bool)
        
        for (cx, cy, ca_deg) in interp_points:
            ca = np.deg2rad(ca_deg)
            cos_a = np.cos(ca)
            sin_a = np.sin(ca)
            
            dx = x_flat - cx
            dy = y_flat - cy
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            in_corridor = in_arm1 | in_arm2
            
            mask &= in_corridor
            
            if not np.any(mask):
                return 0.0
        
        count = np.sum(mask)
        return count * 0.0001

    def optimize_phase(n_ctrl, initial_params, angles):
        def objective(params):
            if time.time() - start_time > time_limit:
                return 0.0 # Bad score (score 0)
            
            # Reshape params to path
            path = []
            for i in range(n_ctrl):
                x = params[2*i]
                y = params[2*i+1]
                a = angles[i]
                path.append((x, y, float(a)))
            
            score = evaluate_path(path)
            return -score
        
        try:
            res = opt.minimize(
                objective, 
                initial_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            # Check if we found a better solution
            # res.fun is the objective value (-score)
            # If res.fun is 0, it means score was 0 or time out.
            # We should compare with the score of initial_params
            init_score = evaluate_path(
                [(initial_params[2*i], initial_params[2*i+1], float(angles[i])) for i in range(n_ctrl)]
            )
            
            if -res.fun > init_score:
                return res.x, -res.fun
            else:
                return initial_params, init_score
        except Exception:
            init_score = evaluate_path(
                [(initial_params[2*i], initial_params[2*i+1], float(angles[i])) for i in range(n_ctrl)]
            )
            return initial_params, init_score

    # Phase 1: N=10
    n_ctrl1 = 10
    angles1 = np.linspace(0, 90, n_ctrl1 + 1)[1:-1]
    init_params1 = np.array([
        [-2.0 * (a / 90.0), 0.0] for a in angles1
    ]).flatten()
    
    params1, score1 = optimize_phase(n_ctrl1, init_params1, angles1)
    best_score = score1
    best_params = params1
    best_n = n_ctrl1
    best_angles = angles1
    
    # Phase 2: N=20
    if time.time() - start_time < time_limit - 150:
        n_ctrl2 = 20
        angles2 = np.linspace(0, 90, n_ctrl2 + 1)[1:-1]
        
        # Interpolate previous path to initialize
        init_params2 = []
        for a in angles2:
            x_val = np.interp(a, best_angles, best_params[0::2])
            y_val = np.interp(a, best_angles, best_params[1::2])
            init_params2.append([x_val, y_val])
        init_params2 = np.array(init_params2).flatten()
        
        params2, score2 = optimize_phase(n_ctrl2, init_params2, angles2)
        
        if score2 > best_score:
            best_score = score2
            best_params = params2
            best_n = n_ctrl2
            best_angles = angles2

    # Phase 3: N=30
    if time.time() - start_time < time_limit - 60:
        n_ctrl3 = 30
        angles3 = np.linspace(0, 90, n_ctrl3 + 1)[1:-1]
        
        init_params3 = []
        for a in angles3:
            x_val = np.interp(a, best_angles, best_params[0::2])
            y_val = np.interp(a, best_angles, best_params[1::2])
            init_params3.append([x_val, y_val])
        init_params3 = np.array(init_params3).flatten()
        
        params3, score3 = optimize_phase(n_ctrl3, init_params3, angles3)
        
        if score3 > best_score:
            best_score = score3
            best_params = params3
            best_n = n_ctrl3
            best_angles = angles3

    # Final Polish: Random perturbations around best solution
    if time.time() - start_time < time_limit - 10:
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            # Perturb
            new_params = best_params + np.random.normal(0, 0.1, size=best_params.shape)
            path = []
            for i in range(best_n):
                x = new_params[2*i]
                y = new_params[2*i+1]
                a = best_angles[i]
                path.append((x, y, float(a)))
            
            score = evaluate_path(path)
            if score > best_score:
                best_score = score
                best_params = new_params

    # Construct final output
    final_path = []
    for i in range(best_n):
        x = best_params[2*i]
        y = best_params[2*i+1]
        a = best_angles[i]
        final_path.append([x, y, a])
        
    if not final_path:
        return np.array([[-1.0, 0.0, 45.0]])
        
    return np.array(final_path)
