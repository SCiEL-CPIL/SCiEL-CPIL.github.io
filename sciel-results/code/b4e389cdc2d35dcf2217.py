# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for moving a sofa through an L-shaped corridor.
    Returns a float array of shape (n, 3) where each row is (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Keep margin from 630s

    # Grid points for sofa area calculation
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    gx, gy = gu.ravel(), gv.ravel()
    N = len(gx)

    def get_dense_path_from_spline(xs, ys, angles, n_samples=200):
        """
        Create a smooth path using cubic splines for x(t), y(t), a(t).
        t goes from 0 to 1.
        xs, ys, angles: arrays of control point coordinates and angles.
        """
        # Create spline interpolators
        t_ctrl = np.linspace(0, 1, len(xs))
        cs_x = CubicSpline(t_ctrl, xs)
        cs_y = CubicSpline(t_ctrl, ys)
        cs_a = CubicSpline(t_ctrl, angles)
        
        # Sample the spline at n_samples points
        t_samples = np.linspace(0, 1, n_samples)
        x_samples = cs_x(t_samples)
        y_samples = cs_y(t_samples)
        a_samples = cs_a(t_samples)
        
        # Create waypoints array
        waypoints = np.column_stack((x_samples, y_samples, a_samples))
        
        # Prepend (0,0,0) and append (-2,0,90) if not already there
        # The spline should start at (0,0,0) and end at (-2,0,90)
        # But let's ensure by adding them explicitly
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        
        full = np.vstack(([start], waypoints, [end]))
        
        # Now densify with linear interpolation between consecutive points
        dense_points = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            if m > 1:
                ts = np.linspace(1.0 / m, 1.0 - 1.0 / m, m - 1)
                new_pts = p1 + ts[:, np.newaxis] * (p2 - p1)
                dense_points.extend(new_pts)
            
            dense_points.append(p2)
        
        return np.array(dense_points)

    def compute_score(dense):
        """
        Compute the raw score (area).
        """
        n_dense = len(dense)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        inside_all = np.ones(N, dtype=bool)
        
        # Batch processing to manage memory
        batch_size = 50
        
        for batch_start in range(0, n_dense, batch_size):
            if time.time() > deadline:
                return 0.0 # Timeout guard
                
            batch_end = min(batch_start + batch_size, n_dense)
            b_cos = cos_a[batch_start:batch_end]
            b_sin = sin_a[batch_start:batch_end]
            b_x = dense[batch_start:batch_end, 0]
            b_y = dense[batch_start:batch_end, 1]
            
            # Vectorized rotation and check
            wx = gx - b_x[:, np.newaxis]
            wy = gy - b_y[:, np.newaxis]
            
            ul = b_cos[:, np.newaxis] * wx + b_sin[:, np.newaxis] * wy
            vl = -b_sin[:, np.newaxis] * wx + b_cos[:, np.newaxis] * wy
            
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            in_base = in_r1 | in_r2
            
            inside_all &= in_base.all(axis=0)
            
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001

    # Define the structure: use cubic spline with k control points
    k = 6  # Number of control points (including endpoints)
    
    # Fixed endpoints
    x_fixed = np.array([0.0, -2.0])
    y_fixed = np.array([0.0, 0.0])
    a_fixed = np.array([0.0, 90.0])
    
    # Free control points: k-2 interior points
    n_free = k - 2
    
    def objective(params):
        """
        params: flat array of x1, y1, x2, y2, ..., x_{k-2}, y_{k-2}
        """
        if time.time() > deadline:
            return 1e6 # Large penalty for timeout
        
        # Reshape params to control points
        x_coords = params[0::2]
        y_coords = params[1::2]
        
        # Construct full control point arrays
        xs = np.concatenate(([0.0], x_coords, [-2.0]))
        ys = np.concatenate(([0.0], y_coords, [0.0]))
        angles = np.linspace(0.0, 90.0, k)  # Angles are evenly spaced
        
        # Ensure finite values
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            return 1e6
            
        # Create dense path from spline
        dense = get_dense_path_from_spline(xs, ys, angles, n_samples=100)
        
        score = compute_score(dense)
        
        # We want to maximize score, so minimize -score
        return -score

    # Initial guesses
    best_score = -1e9
    best_params = None
    
    n_restarts = 15
    rng = np.random.default_rng(42)
    
    # Define bounds for x and y to keep the path reasonable
    bounds = [(-4.0, 2.0), (-4.0, 4.0)] * n_free
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            # Start with a linear path
            init_params = np.zeros(2 * n_free)
            for j in range(n_free):
                t = (j + 1) / (k - 1)
                init_params[2*j] = 0.0 + t * (-2.0 - 0.0)
                init_params[2*j+1] = 0.0
        else:
            # Random initialization within bounds
            init_params = rng.uniform([-4.0, -4.0] * n_free, [2.0, 4.0] * n_free)
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            continue
            
    # Construct the final path from best_params
    if best_params is None:
        # Fallback to a simple linear path
        best_params = np.zeros(2 * n_free)
        for j in range(n_free):
            t = (j + 1) / (k - 1)
            best_params[2*j] = -2.0 * t
            best_params[2*j+1] = 0.0
            
    x_final = best_params[0::2]
    y_final = best_params[1::2]
    
    # Create final waypoints from spline
    xs = np.concatenate(([0.0], x_final, [-2.0]))
    ys = np.concatenate(([0.0], y_final, [0.0]))
    angles = np.linspace(0.0, 90.0, k)
    
    # Sample the spline at reasonable resolution for output
    dense = get_dense_path_from_spline(xs, ys, angles, n_samples=50)
    
    # Remove the explicit start and end points we added, keep only interior waypoints
    # The evaluator will add (0,0,0) and (-2,0,90)
    interior_waypoints = dense[1:-1]
    
    return interior_waypoints

# EVOLVE-BLOCK-END
