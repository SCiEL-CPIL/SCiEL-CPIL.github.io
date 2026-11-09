import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill

    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]

    def get_base_mask(u, v):
        # Base corridor:
        # Arm 1: 0 <= u <= 1, -3 <= v <= 1
        # Arm 2: -3 <= u <= 1, 0 <= v <= 1
        return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
               ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

    def evaluate_path(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        """
        # Ensure path is a numpy array
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            
        # Prepend and append fixed endpoints
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
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = get_base_mask(u, v)
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0
                
        return np.sum(valid) * 0.0001

    # Strategy:
    # The current program uses a fixed set of 17 control points (every 5 degrees) and optimizes their (x,y) coordinates.
    # This new strategy uses a DENSER set of control points (every 2 degrees) to allow for a more precise path,
    # but employs a two-stage optimization to handle the increased dimensionality (88 variables vs 34).
    # Stage 1: Optimize a lower-dimensional parameterization (e.g., 5-6 points) to get a good initial guess.
    # Stage 2: Use that guess to initialize the high-resolution optimization with a limited number of iterations.
    
    # Stage 1: Coarse optimization with 5 points (10, 30, 50, 70, 80 degrees? No, let's stick to 10, 30, 50, 70)
    # Actually, let's use 4 points: 20, 40, 60, 80.
    angles_coarse = np.array([20.0, 40.0, 60.0, 80.0])
    n_coarse = len(angles_coarse)
    
    def objective_coarse(params):
        if time.time() > deadline - 100:
            return 0.0
        path = np.zeros((n_coarse, 3))
        for i in range(n_coarse):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = angles_coarse[i]
        return -evaluate_path(path)

    # Initial guess for coarse
    x0_coarse = np.zeros(2 * n_coarse)
    for i, a in enumerate(angles_coarse):
        t = a / 90.0
        x0_coarse[2*i] = -2 * t
        x0_coarse[2*i+1] = math.sin(math.pi * t) * 0.8 # Slightly lower amplitude
        
    best_coarse = x0_coarse
    try:
        res_coarse = minimize(
            objective_coarse, 
            x0_coarse, 
            method='Nelder-Mead', 
            options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-4, 'disp': False}
        )
        if res_coarse.fun < -1e-10: # Ensure it's valid
            best_coarse = res_coarse.x
    except Exception:
        pass

    # Stage 2: Fine optimization with 44 points (every 2 degrees: 2, 4, ..., 88)
    angles_fine = np.arange(2, 90, 2)
    n_fine = len(angles_fine)
    
    # Interpolate the coarse solution to the fine grid to initialize
    x0_fine = np.zeros(2 * n_fine)
    for i, a in enumerate(angles_fine):
        # Find nearest coarse points
        # Simple linear interpolation between coarse points
        # Coarse angles: 20, 40, 60, 80
        # We need to map 'a' to (x, y)
        # If a < 20, extrapolate or use endpoint?
        # Let's just use the coarse values directly if close, or interpolate.
        
        # For simplicity, let's just use the coarse optimization result as a guide
        # and add a small random perturbation? No, deterministic is better.
        
        # Let's interpolate the coarse path onto the fine angles.
        # Coarse path:
        coarse_path = np.zeros((n_coarse, 2))
        for j in range(n_coarse):
            coarse_path[j, 0] = best_coarse[2*j]
            coarse_path[j, 1] = best_coarse[2*j+1]
            
        # Interpolate
        x_interp = np.interp(a, angles_coarse, coarse_path[:, 0])
        y_interp = np.interp(a, angles_coarse, coarse_path[:, 1])
        
        x0_fine[2*i] = x_interp
        x0_fine[2*i+1] = y_interp

    def objective_fine(params):
        if time.time() > deadline - 10:
            return 0.0
        path = np.zeros((n_fine, 3))
        for i in range(n_fine):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = angles_fine[i]
        return -evaluate_path(path)

    best_fine = x0_fine
    best_fine_score = -np.inf
    
    try:
        # Use a limited number of iterations for the fine stage
        # 88 variables is a lot for Nelder-Mead. 
        # Let's try COBYLA or just a short Nelder-Mead run.
        # Nelder-Mead might struggle with 88 dims.
        # Let's try to reduce the dimensionality by using a smooth basis?
        # Or just stick with Nelder-Mead but limit iterations.
        
        res_fine = minimize(
            objective_fine, 
            x0_fine, 
            method='Nelder-Mead', 
            options={'maxiter': 50, 'xatol': 1e-2, 'fatol': 1e-3, 'disp': False}
        )
        
        if time.time() < deadline - 5:
            score = -res_fine.fun
            if score > best_fine_score:
                best_fine_score = score
                best_fine = res_fine.x
                
    except Exception:
        pass

    # Construct final path from the best fine result
    final_path = np.zeros((n_fine, 3))
    for i in range(n_fine):
        final_path[i, 0] = best_fine[2*i]
        final_path[i, 1] = best_fine[2*i+1]
        final_path[i, 2] = angles_fine[i]
        
    return final_path
