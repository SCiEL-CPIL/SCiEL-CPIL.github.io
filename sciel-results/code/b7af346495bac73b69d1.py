import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 40.0

    # Sofa points definition
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x, P_y = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x = P_x.flatten()
    P_y = P_y.flatten()
    N_pts = len(P_x)

    def evaluate_path(path):
        # Densify the path as the evaluator does
        full_path = [(0.0, 0.0, 0.0)] + path + [(-2.0, 0.0, 90.0)]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            x1, y1, a1 = full_path[i]
            x2, y2, a2 = full_path[i+1]
            
            da = abs(a2 - a1)
            m = max(1, int(np.ceil(da)))
            
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    a = a1 + t * (a2 - a1)
                    densified.append((x, y, a))
            densified.append(full_path[i+1])
        
        # Remove duplicates just in case (though interpolation usually handles it)
        # Actually, the loop above adds full_path[i+1] every time, so the last point is added multiple times?
        # No, the loop runs for i in 0..len-2.
        # i=0: adds interp, adds full_path[1]
        # i=1: adds interp, adds full_path[2]
        # ...
        # i=len-2: adds interp, adds full_path[-1]
        # So densified has full_path[0] + (interps + full_path[i+1] for each step).
        # This is correct.
        
        in_sofa = np.ones(N_pts, dtype=bool)
        
        for x, y, a in densified:
            theta = np.deg2rad(a)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            dx = P_x - x
            dy = P_y - y
            
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = cond1 | cond2
            
            in_sofa &= mask
            
            if not np.any(in_sofa):
                return 0.0
        
        return np.sum(in_sofa) * 0.0001

    # Parameterization
    # Use a variable number of angles? Or fixed?
    # Fixed is easier for optimizers.
    # Let's try 15 angles for better resolution.
    # The angles should be spread out to allow smooth motion.
    angles = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 88.0])
    n_angles = len(angles)
    n_params = 2 * n_angles

    # Initial guess
    # Based on the previous good path, interpolated to our new angles
    base_x = [-0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7]
    base_y = [0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3]
    base_a = [15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0]
    
    # Extrapolate/Interpolate
    init_x = np.interp(angles, base_a, base_x, left=base_x[0], right=base_x[-1])
    init_y = np.interp(angles, base_a, base_y, left=base_y[0], right=base_y[-1])
    
    p0 = np.concatenate([init_x, init_y])
    
    def objective(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        
        path = []
        for i in range(n_angles):
            path.append([x_vals[i], y_vals[i], angles[i]])
        
        area = evaluate_path(path)
        return -area

    best_params = p0
    best_val = objective(best_params)
    
    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Optimization Strategy:
    # 1. Powell for global-ish search
    # 2. COBYLA for local refinement
    
    try:
        # Powell
        res = so.minimize(objective, best_params, method='Powell', 
                          options={'maxiter': 3000, 'maxfev': 3000, 'disp': False})
        if not time_guard():
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass

    if not time_guard():
        try:
            # COBYLA
            res2 = so.minimize(objective, best_params, method='COBYLA', 
                               options={'maxiter': 1000, 'maxfev': 1000, 'disp': False})
            if not time_guard():
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x
        except Exception:
            pass
            
    # Local Nelder-Mead with small step size
    if not time_guard():
        try:
            res3 = so.minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'maxfev': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            if not time_guard():
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x
        except Exception:
            pass

    # If time remains, try restarting with random perturbations
    if not time_guard():
        np.random.seed(42)
        for _ in range(20):
            if time_guard():
                break
            perturbed = best_params + np.random.normal(0, 0.1, size=n_params)
            # Clip to reasonable bounds?
            # x in [-2, 0], y in [0, 1] roughly
            perturbed[:n_angles] = np.clip(perturbed[:n_angles], -2.0, 0.0)
            perturbed[n_angles:] = np.clip(perturbed[n_angles:], 0.0, 1.0)
            
            val = objective(perturbed)
            if val < best_val:
                best_val = val
                best_params = perturbed
                
                # Quick local refinement
                try:
                    res_r = so.minimize(objective, best_params, method='Nelder-Mead', 
                                        options={'maxiter': 100, 'maxfev': 100, 'disp': False})
                    if not time_guard():
                        if res_r.fun < best_val:
                            best_val = res_r.fun
                            best_params = res_r.x
                except Exception:
                    pass

    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]
    
    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])
        
    path_arr = np.array(final_path, dtype=np.float64)
    
    return path_arr
