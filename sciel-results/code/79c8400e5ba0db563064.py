import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill

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
    # We control the waypoints. The evaluator inserts linear interpolation.
    # We want to maximize the intersection of the moving corridor.
    
    # Let's use a smooth parameterization of the path.
    # The path goes from angle 0 to 90.
    # Let's parameterize by angle a in [0, 90].
    # We choose a set of control points to optimize.
    # To allow high resolution, let's optimize the path at every 5 degrees.
    # Waypoints at a = 5, 10, ..., 85. (17 points)
    # This gives us 34 variables (x,y for each).
    # The evaluator will interpolate between 0-5, 5-10, ..., 85-90.
    
    angles_ctrl = np.arange(5, 90, 5)
    n_ctrl = len(angles_ctrl)
    
    def objective(params):
        if time.time() > deadline - 5:
            return 10.0 # Penalize if too late, we want to minimize negative score
            
        # params: x1, y1, x2, y2, ...
        path = np.zeros((n_ctrl, 3))
        for i in range(n_ctrl):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = angles_ctrl[i]
            
        score = evaluate_path(path)
        return -score

    # Initial guess:
    # A reasonable guess for the moving sofa path in an L-shaped corridor.
    # The sofa rotates from 0 to 90 degrees.
    # The center of rotation effectively moves.
    # Let's try a path that stays near the "inner" corner of the L.
    # The L is at (0,0) effectively? No, the corridor moves.
    # At a=0, the corridor is fixed at (0,0).
    # At a=90, the corridor is fixed at (-2,0) rotated 90.
    
    # Let's start with a simple curve:
    # x(a) = -a/45  (linear from 0 to -2)
    # y(a) = 1.5 * sin(pi * a / 90) (arc upwards)
    
    x0 = np.zeros(2 * n_ctrl)
    for i, a in enumerate(angles_ctrl):
        t = a / 90.0
        x0[2*i] = -2.0 * t
        x0[2*i+1] = 1.2 * math.sin(math.pi * t)
        
    # Refine using Nelder-Mead
    from scipy.optimize import minimize
    
    best_result = None
    best_score = -np.inf
    
    # Phase 1: Coarse optimization
    try:
        result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-4, 'disp': False}
        )
        if time.time() < deadline - 20:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_result = result.x
    except Exception as e:
        pass
        
    # Phase 2: Fine optimization with more points?
    # Let's try adding more control points if time permits.
    # Or just run Nelder-Mead again with tighter tolerances on the current best.
    
    if best_result is not None:
        try:
            result2 = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6, 'disp': False}
            )
            score2 = -result2.fun
            if score2 > best_score:
                best_score = score2
                best_result = result2.x
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to initial guess if optimization failed or timed out
        best_result = x0
        # We should try to at least get a valid score for the initial guess
        # but we already have x0 which is a valid path.
        
    # Construct final path
    final_path = np.zeros((n_ctrl, 3))
    for i in range(n_ctrl):
        final_path[i, 0] = best_result[2*i]
        final_path[i, 1] = best_result[2*i+1]
        final_path[i, 2] = angles_ctrl[i]
        
    return final_path
