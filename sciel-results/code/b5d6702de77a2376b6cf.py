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

    def evaluate_path_dense(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        """
        # Prepend (0,0,0) and append (-2,0,90)
        # The evaluator does this, so we must simulate it for accurate scoring.
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

    def evaluate_path_fast(params):
        """
        Evaluate the path defined by quadratic Bezier parameters.
        params: [x1, y1, x2, y2]
        x(t) = (1-t)^2 * 0 + 2(1-t)t * x1 + t^2 * x2  (Start 0, End -2)
        y(t) = (1-t)^2 * 0 + 2(1-t)t * y1 + t^2 * y2  (Start 0, End 0)
        
        We sample the path at every integer degree to ensure the evaluator's
        interpolation matches our intent.
        """
        if time.time() > deadline - 5:
            return 0.0
        
        x1, y1, x2, y2 = params
        
        # Generate points at integer degrees 1..89
        angles = np.arange(1, 90)
        t = angles / 90.0
        
        # Quadratic Bezier
        # B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
        # P0 = (0,0), P2 = (-2,0)
        # P1 = (x1, y1), P2_ctrl = (x2, y2) ? 
        # Wait, standard quadratic Bezier has 3 control points: P0, P1, P2.
        # Here P0=(0,0), P2=(-2,0) are fixed. P1 is the single control point.
        # But we have 4 params. Let's use a cubic Bezier? 
        # Cubic: P0=(0,0), P3=(-2,0). P1=(x1,y1), P2=(x2,y2).
        # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        
        one_minus_t = 1.0 - t
        
        bx = 3 * (one_minus_t**2) * t * x1 + 3 * (one_minus_t) * (t**2) * x2 + (t**3) * (-2.0)
        by = 3 * (one_minus_t**2) * t * y1 + 3 * (one_minus_t) * (t**2) * y2 + (t**3) * 0.0
        
        path = np.column_stack((bx, by, angles))
        
        return evaluate_path_dense(path)

    # Initial guess for Cubic Bezier control points P1, P2
    # The path should bulge outwards to clear the corner.
    # P1 is near the start, P2 is near the end.
    # Start (0,0), End (-2,0).
    # A good initial guess might be to have the curve go "up" in y.
    # Let's try P1 = (-0.5, 1.0), P2 = (-1.5, 1.0)
    x0 = np.array([-0.5, 1.0, -1.5, 1.0])
    
    from scipy.optimize import minimize
    
    best_result = None
    best_score = -np.inf
    
    # We can run multiple restarts if time permits
    # But first, let's just run one robust optimization.
    
    try:
        # Use Nelder-Mead for low-dimensional optimization
        result = minimize(
            evaluate_path_fast, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False}
        )
        
        if time.time() < deadline - 10:
            score = result.fun
            if score > best_score:
                best_score = score
                best_result = result.x
                
    except Exception as e:
        pass
        
    if best_result is None:
        best_result = x0
        
    # Construct final path from the best Bezier parameters
    x1, y1, x2, y2 = best_result
    
    # Generate the output points.
    # The evaluator will interpolate between these.
    # To be safe and precise, we output points at every integer degree 1..89.
    # This ensures m=1 for all segments, so no extra interpolation by the evaluator.
    angles = np.arange(1, 90)
    t = angles / 90.0
    one_minus_t = 1.0 - t
    
    bx = 3 * (one_minus_t**2) * t * x1 + 3 * (one_minus_t) * (t**2) * x2 + (t**3) * (-2.0)
    by = 3 * (one_minus_t**2) * t * y1 + 3 * (one_minus_t) * (t**2) * y2 + (t**3) * 0.0
    
    final_path = np.column_stack((bx, by, angles))
    
    return final_path
