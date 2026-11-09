import numpy as np
import time
from scipy.optimize import differential_evolution

def moving_sofa_2d():
    # Grid setup
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Precompute start and end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Fixed intermediate angles
    fixed_thetas = [15.0, 30.0, 45.0, 60.0, 75.0]
    k = len(fixed_thetas)
    
    def objective(params):
        # params: [x1, y1, x2, y2, ..., xk, yk]
        # Construct intermediate points
        inter_points = []
        for i in range(k):
            x = params[2*i]
            y = params[2*i+1]
            th = fixed_thetas[i]
            inter_points.append([x, y, th])
        
        # Full path
        path = [start] + inter_points + [end]
        
        # Interpolate to 1-degree steps
        # We need to generate the dense path
        dense_path = []
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            a1 = p1[2]
            a2 = p2[2]
            # m = max(1, ceil(|a2-a1|))
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for j in range(m+1):
                if i == len(path)-2 and j == m:
                    continue # Skip last point of last segment to avoid duplicate with next segment's start? 
                    # Actually, evaluator adds linear interpolations between consecutive rows.
                    # "Between consecutive rows it adds linear interpolations ... in m equal steps"
                    # This usually means points at t=0, 1/m, ..., 1.
                    # If we do this for every segment, we might duplicate points at the joints.
                    # Duplicates don't hurt the AND mask.
                    t = j / m
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    a = a1 + t * (a2 - a1)
                    dense_path.append([x, y, a])
        
        if not dense_path:
            return 0.0
            
        # Compute intersection
        # For each dense point, compute mask
        # Mask shape: (N_g,)
        # We want AND of all masks.
        # Start with all True
        sofa_mask = np.ones(len(grid_points), dtype=bool)
        
        for p in dense_path:
            x, y, a_deg = p
            a_rad = np.deg2rad(a_deg)
            # Rotate grid points by -a_rad and translate by (-x, -y)
            # w' = R(-a)(w - c)
            # u = w - c
            # u_x = X - x, u_y = Y - y
            # v_x = cos(a)*u_x + sin(a)*u_y  (Rotation by -a: [cos a, sin a; -sin a, cos a] ? No.)
            # R(theta) = [cos, -sin; sin, cos]
            # R(-theta) = [cos, sin; -sin, cos]
            # So u'_x = cos(a)*u_x + sin(a)*u_y
            # u'_y = -sin(a)*u_x + cos(a)*u_y
            
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            u_x = X.ravel() - x
            u_y = Y.ravel() - y
            
            v_x = cos_a * u_x + sin_a * u_y
            v_y = -sin_a * u_x + cos_a * u_y
            
            # Base corridor:
            # 0 <= v_x <= 1 and -3 <= v_y <= 1
            # OR
            # -3 <= v_x <= 1 and 0 <= v_y <= 1
            
            mask1 = (v_x >= 0) & (v_x <= 1) & (v_y >= -3) & (v_y <= 1)
            mask2 = (v_x >= -3) & (v_x <= 1) & (v_y >= 0) & (v_y <= 1)
            
            current_mask = mask1 | mask2
            
            sofa_mask &= current_mask
            
            # Early exit if empty
            if not np.any(sofa_mask):
                return 0.0
                
        return np.sum(sofa_mask) * 0.0001

    # Optimization
    # Bounds for x, y
    # Reasonable bounds: x in [-2, 1], y in [-1, 1]
    bounds = [(-2.0, 1.0), (-1.0, 1.0)] * k
    
    start_time = time.time()
    timeout = 500 # Leave margin
    
    try:
        # Use differential evolution
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            tol=1e-4,
            seed=42,
            polish=True
        )
    except Exception:
        # Fallback to a heuristic path if optimization fails or times out badly
        result = None

    if result and result.success and result.fun > 0:
        best_params = result.x
    else:
        # Fallback: Use the seed program's path or a simple heuristic
        # Let's just return a reasonable static path
        best_params = np.array([
            -0.1, 0.1,
            -0.2, 0.2,
            -0.3, 0.3,
            -0.4, 0.3,
            -0.5, 0.2
        ])
        
    # Construct final path from best_params
    inter_points = []
    for i in range(k):
        x = best_params[2*i]
        y = best_params[2*i+1]
        th = fixed_thetas[i]
        inter_points.append([x, y, th])
        
    path = np.array(inter_points)
    return path
