import numpy as np
from scipy.optimize import minimize
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    
    # Grid points
    # i = 0..449, j = 0..299
    # x_grid = -3 + 0.01 * i
    # y_grid = -1.5 + 0.01 * j
    i_vals = np.arange(-3, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(i_vals, j_vals, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Base corridor check function
    # Point (u,v) is in base if:
    # (0 <= u <= 1 and -3 <= v <= 1) OR (-3 <= u <= 1 and 0 <= v <= 1)
    
    def in_base(points):
        # points: (N, 2)
        u, v = points[:, 0], points[:, 1]
        mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return mask1 | mask2

    def get_full_path(path):
        # path: (n, 3)
        # Prepend (0,0,0) and append (-2,0,90)
        start = np.array([[0.0, 0.0, 0.0]])
        end = np.array([[-2.0, 0.0, 90.0]])
        full = np.vstack([start, path, end])
        
        # Interpolate
        # The evaluator adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps.
        # This means if angle diff is 15, it adds 14 intermediate points? 
        # "in m = ... steps" usually means m intervals, so m+1 points total including endpoints?
        # Or m steps means m segments?
        # "linear interpolations ... in m equal steps"
        # If a2-a1 = 15, m=15. 
        # Typically this means the segment is divided into 15 parts, so 14 internal points.
        # Let's assume the evaluator constructs a dense path.
        # I need to replicate this to evaluate correctly.
        
        dense_path = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            # t from 0 to 1 in m steps?
            # If m=1, just p2?
            # "adds linear interpolations ... in m equal steps"
            # This phrasing is slightly ambiguous. 
            # If I move from A to B in 1 step, I just go to B.
            # If I move in 2 steps, I go to midpoint, then B.
            # So number of segments = m.
            ts = np.linspace(0, 1, m+1)
            for t in ts[1:]: # Skip 0 as p1 is already added
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                pa = p1[2] + t * (p2[2] - p1[2])
                dense_path.append([px, py, pa])
        return np.array(dense_path)

    def evaluate_path(path):
        # path: (n, 3)
        dense = get_full_path(path)
        
        # For each placement, transform grid points and check if in base.
        # Intersection: point must be in base for ALL placements.
        # Initialize mask as True for all grid points
        mask = np.ones(len(grid_points), dtype=bool)
        
        for i in range(len(dense)):
            x, y, a_deg = dense[i]
            a_rad = -np.deg2rad(a_deg) # R(-a)
            c = np.cos(a_rad)
            s = np.sin(a_rad)
            
            # Translate
            dx = grid_points[:, 0] - x
            dy = grid_points[:, 1] - y
            
            # Rotate
            u = c * dx - s * dy
            v = s * dx + c * dy
            
            pts = np.column_stack((u, v))
            in_b = in_base(pts)
            
            mask &= in_b
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask)

    # Initial path from baseline
    path0 = np.array(
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
    
    best_path = path0.copy()
    best_score = evaluate_path(best_path)
    
    # Optimization
    # Flatten path
    x0 = best_path.flatten()
    
    def objective(params):
        p = params.reshape(-1, 3)
        # Ensure angles are sorted and within (0, 90)
        # If not, return bad score
        angles = p[:, 2]
        if np.any(angles <= 0) or np.any(angles >= 90):
            return 100000
        if not np.all(np.diff(angles) > 0):
            return 100000
            
        score = evaluate_path(p)
        return -score

    # Use Nelder-Mead
    # To save time, use a subset of grid points for optimization?
    # Let's try full grid first, but limit iterations.
    
    try:
        # Run a few local optimizations
        for _ in range(5):
            if time.time() > deadline - 10:
                break
                
            # Small perturbation
            noise = np.random.normal(0, 0.1, x0.shape)
            p_init = x0 + noise
            # Ensure angles are valid
            # Sort by angle? The order of points in the path matters for interpolation.
            # The optimizer might mess up the order.
            # To be safe, I will optimize x,y and keep angles fixed or optimize them with constraints.
            # Given the complexity, let's just optimize x,y for the fixed angles of the baseline.
            # This is a safer, lower-dimensional search.
            
            fixed_angles = best_path[:, 2]
            
            def objective_xy(params):
                p = params.copy()
                p[2::3] = fixed_angles # Restore angles
                p = p.reshape(-1, 3)
                return objective(p)
                
            res = minimize(objective_xy, x0[0::3], method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < -best_score:
                # res.x contains only x,y coordinates
                new_path = best_path.copy()
                new_path[:, 0] = res.x[0::2]
                new_path[:, 1] = res.x[1::2]
                new_score = evaluate_path(new_path)
                if new_score > best_score:
                    best_path = new_path
                    best_score = new_score
                    x0 = best_path.flatten()
                    
    except Exception as e:
        pass

    return best_path
