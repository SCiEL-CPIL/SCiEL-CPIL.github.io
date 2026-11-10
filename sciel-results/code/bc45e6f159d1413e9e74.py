# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Flatten for easier vectorization
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Precompute base mask logic
    # Base is union of R1 and R2.
    # R1: 0 <= u <= 1, -3 <= v <= 1
    # R2: -3 <= u <= 1, 0 <= v <= 1
    
    def check_inside(x, y, a_deg):
        # a_deg in degrees
        # w = (X_flat - x, Y_flat - y)
        # w_rot = R(-a) * w
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        # Rotation
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        # Check R1 and R2
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        inside = in_r1 | in_r2
        return inside
    
    def evaluate_path(path):
        # path is array of (x, y, a)
        # Densify as per evaluator:
        # "Between consecutive rows it adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps"
        # Evaluator adds (0,0,0) and (-2,0,90).
        
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(n_points, dtype=bool)
        
        # Process each segment
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            # Interpolate
            ts = np.linspace(0, 1, m+1)
            
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                
                inside = check_inside(x, y, a)
                mask &= inside
                
                # Early termination: if mask is all False, break early
                if not mask.any():
                    return 0.0

        return np.sum(mask) * 0.0001
    
    # Optimization
    # Define initial path based on seed
    seed_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_path = seed_path.copy()
    best_score = evaluate_path(best_path)
    
    start_time = time.time()
    time_limit = 590.0 # Safety margin
    
    # Define parameters to optimize
    # We'll optimize the control points.
    # Let's use 6 control points at a = 15, 30, 45, 60, 75, 85.
    # We interpolate x,y linearly between them.
    
    def make_path_from_params(params):
        # params: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6]
        angles = [15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
        path = []
        for i in range(6):
            x = params[2*i]
            y = params[2*i+1]
            a = angles[i]
            path.append([x, y, a])
        return np.array(path, dtype=np.float64)
    
    # Initial params from seed
    # 15: -0.1, 0.1
    # 30: interpolate between 20 and 34? 20->[-0.2,0.2], 34->[-0.3,0.3]. Mid 27->[-0.25,0.25]
    # 45: interpolate between 40 and 50? 40->[-0.3,0.3], 50->[-0.4,0.3]. Mid 45->[-0.35,0.3]
    # 60: interpolate between 55 and 75? 55->[-0.5,0.2], 75->[-0.6,0.2]. Mid 65->[-0.55,0.2]
    # 75: -0.6, 0.2
    # 85: interpolate between 75 and 86? 75->[-0.6,0.2], 86->[-0.7,0.3]. Mid 80.5->[-0.65,0.25]
    current_params = np.array([-0.1, 0.1, -0.25, 0.25, -0.35, 0.3, -0.55, 0.2, -0.6, 0.2, -0.65, 0.25])
    
    # Objective function for optimization (minimize negative score)
    def objective(params):
        path = make_path_from_params(params)
        val = evaluate_path(path)
        return -val
    
    # Use Nelder-Mead optimization
    best_params = current_params.copy()
    best_val = best_score
    
    # Multiple starting points
    start_points = [
        current_params,
        np.array([-0.05, 0.05, -0.2, 0.2, -0.3, 0.25, -0.5, 0.15, -0.55, 0.15, -0.6, 0.2]),
        np.array([-0.15, 0.15, -0.3, 0.3, -0.4, 0.35, -0.6, 0.25, -0.65, 0.25, -0.7, 0.3]),
        np.array([-0.0, 0.0, -0.15, 0.15, -0.25, 0.2, -0.45, 0.1, -0.5, 0.1, -0.55, 0.15]),
    ]
    
    for sp in start_points:
        if time.time() - start_time > time_limit:
            break
        
        result = minimize(objective, sp, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if -result.fun > best_val:
            best_val = -result.fun
            best_params = result.x
    
    # Local refinement around the best point found
    if time.time() - start_time < time_limit:
        # Fine-tune with smaller steps
        step = 0.01
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            for dim in range(len(best_params)):
                # Positive step
                new_params = best_params.copy()
                new_params[dim] += step
                path = make_path_from_params(new_params)
                val = evaluate_path(path)
                if val > best_val:
                    best_val = val
                    best_params = new_params
                    improved = True
                
                # Negative step
                new_params = best_params.copy()
                new_params[dim] -= step
                path = make_path_from_params(new_params)
                val = evaluate_path(path)
                if val > best_val:
                    best_val = val
                    best_params = new_params
                    improved = True
            
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
    
    final_path = make_path_from_params(best_params)
    
    # Final check: ensure the final path is valid
    final_score = evaluate_path(final_path)
    if final_score > best_score:
        best_score = final_score
        best_path = final_path
    
    return best_path

# EVOLVE-BLOCK-END
