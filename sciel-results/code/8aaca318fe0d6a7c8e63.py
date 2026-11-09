import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill

    # Grid for evaluation, matching the evaluator
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
        # The evaluator puts (0, 0, 0) before and (-2, 0, 90) after.
        # If we provide points at every integer degree 1..89, the interpolation steps are m=1.
        # This means the dense path is exactly: (0,0,0), (x1,y1,1), ..., (x89,y89,89), (-2,0,90).
        # We can optimize x1..x89, y1..y89 directly.
        
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
    # Use a dense path with control points at every integer degree from 1 to 89.
    # This ensures that the linear interpolation between any two consecutive points
    # (including the fixed start and end) is a single step (m=1), so the actual
    # placements evaluated are exactly the ones we specify (plus the endpoints).
    
    angles_ctrl = np.arange(1, 90)
    n_ctrl = len(angles_ctrl) # 89 points
    
    # Initial guess:
    # The "moving sofa" problem solution is known to follow a specific curve.
    # A good initial guess is a smooth curve connecting (0,0) to (-2,0) while rotating.
    # Let's try a simple arc or a combination of linear and sinusoidal.
    # x(t) goes from 0 to -2. y(t) goes up and down.
    # At a=0, x=0, y=0.
    # At a=90, x=-2, y=0.
    # Let's try x = -2 * (a/90) and y = 1.2 * sin(pi * a/90).
    
    x0 = np.zeros(2 * n_ctrl)
    for i, a in enumerate(angles_ctrl):
        t = a / 90.0
        x0[2*i] = -2.0 * t
        x0[2*i+1] = 1.2 * math.sin(math.pi * t)
        
    def build_path(params):
        path = np.zeros((n_ctrl, 3))
        for i in range(n_ctrl):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = angles_ctrl[i]
        return path

    def get_score(params):
        if time.time() > deadline - 10:
            return -1.0
        path = build_path(params)
        return evaluate_path(path)

    # Coordinate Descent Optimization
    # We iterate through each (x, y) pair and try to optimize it locally.
    # Since the objective is non-smooth (grid based), we use a local search.
    
    current_params = x0.copy()
    current_score = get_score(current_params)
    
    # To speed up, we can pre-compute the base mask logic? No, rotation changes u,v.
    # But we can optimize the evaluation loop.
    
    # Let's do a few passes of coordinate descent.
    # For each point i, try small perturbations in x and y.
    # Step size can start large and decrease.
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() > deadline - 30:
            break
            
        improved = True
        while improved:
            improved = False
            for i in range(n_ctrl):
                if time.time() > deadline - 30:
                    break
                
                orig_x = current_params[2*i]
                orig_y = current_params[2*i+1]
                
                best_x = orig_x
                best_y = orig_y
                best_local_score = current_score
                
                # Try perturbations in x
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        test_params = current_params.copy()
                        test_params[2*i] = orig_x + dx
                        test_params[2*i+1] = orig_y + dy
                        
                        s = get_score(test_params)
                        if s > best_local_score:
                            best_local_score = s
                            best_x = orig_x + dx
                            best_y = orig_y + dy
                            
                # Also try pure x and y moves
                for dx in [-step, step]:
                    test_params = current_params.copy()
                    test_params[2*i] = orig_x + dx
                    s = get_score(test_params)
                    if s > best_local_score:
                        best_local_score = s
                        best_x = orig_x + dx
                        best_y = orig_y
                        
                for dy in [-step, step]:
                    test_params = current_params.copy()
                    test_params[2*i+1] = orig_y + dy
                    s = get_score(test_params)
                    if s > best_local_score:
                        best_local_score = s
                        best_x = orig_x
                        best_y = orig_y + dy

                if best_local_score > current_score:
                    current_params[2*i] = best_x
                    current_params[2*i+1] = best_y
                    current_score = best_local_score
                    improved = True

    # Final check and construction
    if time.time() > deadline - 5:
        # Use initial guess if optimization took too long? 
        # No, current_params is at least as good as x0.
        pass
        
    final_path = build_path(current_params)
    
    return final_path
