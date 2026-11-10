import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0 # Hard limit
    safety_margin = 30.0 # Stop 30s before limit

    # Sofa points definition
    # i = 0..449, j = 0..299
    # x = -3 + 0.01*i
    # y = -1.5 + 0.01*j
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x, P_y = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x = P_x.flatten()
    P_y = P_y.flatten()
    N_pts = len(P_x)

    # Pre-compute densified path structure for a given set of waypoints
    # This avoids re-computing the densification logic in every objective call
    # We assume the waypoints are at fixed angles.
    # The evaluator densifies between consecutive waypoints.
    # Since we fix the angles of our waypoints, the densification structure (number of points, t values)
    # is fixed. Only x, y change.
    
    # We will define a helper to build the dense path given x, y arrays for fixed angles.
    # full_path = [(0,0,0)] + [(x_i, y_i, a_i) for i] + [(-2,0,90)]
    
    def build_dense_path_indices(angles):
        """
        Returns a list of tuples (source_idx, t) where source_idx indicates
        which segment the point comes from, and t is the interpolation parameter.
        Actually, it's easier to just store the dense points' x, y as linear combos of waypoints.
        
        Let's pre-compute the "weights" for each dense point in terms of the waypoints.
        Dense point k is on segment i (between waypoint i-1 and i, or fixed endpoints).
        
        Let's just store the dense path structure:
        dense_x = []
        dense_y = []
        dense_a = []
        
        We'll create a list of coefficients.
        Each dense point is a linear interpolation between two "anchor" points.
        Anchors are: Start (0,0,0), Waypoint 0..N-1, End (-2,0,90).
        
        Let's index anchors: 0=Start, 1..N=Waypoints, N+1=End.
        """
        N = len(angles)
        # Anchors:
        # 0: (0,0,0)
        # 1..N: (x_i, y_i, a_i)
        # N+1: (-2,0,90)
        
        dense_points = [] # List of (anchor_start_idx, anchor_end_idx, t)
        
        # Segment 0: Start to Waypoint 0
        a1 = 0.0
        a2 = angles[0]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m > 1:
            for k in range(1, m):
                t = k / m
                dense_points.append((0, 1, t))
        
        # Segments between waypoints
        for i in range(N - 1):
            a1 = angles[i]
            a2 = angles[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    dense_points.append((i + 1, i + 2, t))
        
        # Segment N: Waypoint N-1 to End
        a1 = angles[N-1]
        a2 = 90.0
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m > 1:
            for k in range(1, m):
                t = k / m
                dense_points.append((N, N + 1, t))
        
        return dense_points

    # Function to evaluate the area for a given path (x, y arrays for fixed angles)
    def evaluate_path(x_vals, y_vals, angles):
        # Construct the full set of dense points
        # We need the x, y, a for each dense point.
        # Anchors:
        # 0: (0,0,0)
        # 1..N: (x_vals[i], y_vals[i], angles[i])
        # N+1: (-2,0,90)
        
        N = len(angles)
        
        # Get pre-computed dense point structure
        # Note: build_dense_path_indices depends on angles, which are fixed here.
        # We can cache it. But for simplicity in this closure, let's just compute it once outside?
        # Since angles are fixed for the optimization phase, we can pre-compute.
        
        # Let's assume dense_points_structure is passed in or global.
        # For now, let's just compute the dense points on the fly but efficiently.
        
        # Actually, recomputing m and t every time is cheap. The expensive part is the mask.
        # Let's just build the list of (x, y, a) for dense points.
        
        dense_x = [0.0]
        dense_y = [0.0]
        dense_a = [0.0]
        
        # Add waypoints
        for i in range(N):
            dense_x.append(x_vals[i])
            dense_y.append(y_vals[i])
            dense_a.append(angles[i])
        
        # Add end point
        dense_x.append(-2.0)
        dense_y.append(0.0)
        dense_a.append(90.0)
        
        # Now insert intermediate points
        # This is O(N * max_m). Since N is small (10) and max_m is ~90, this is fine.
        
        full_dense_x = [0.0]
        full_dense_y = [0.0]
        full_dense_a = [0.0]
        
        # Segment 0: 0 to 1
        a1, a2 = dense_a[0], dense_a[1]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m > 1:
            for k in range(1, m):
                t = k / m
                full_dense_x.append(dense_x[0] + t * (dense_x[1] - dense_x[0]))
                full_dense_y.append(dense_y[0] + t * (dense_y[1] - dense_y[0]))
                full_dense_a.append(a1 + t * (a2 - a1))
        
        # Segments 1 to N-1
        for i in range(1, N):
            a1, a2 = dense_a[i], dense_a[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    full_dense_x.append(dense_x[i] + t * (dense_x[i+1] - dense_x[i]))
                    full_dense_y.append(dense_y[i] + t * (dense_y[i+1] - dense_y[i]))
                    full_dense_a.append(a1 + t * (a2 - a1))
            # Append the end of this segment (which is the start of the next)
            full_dense_x.append(dense_x[i+1])
            full_dense_y.append(dense_y[i+1])
            full_dense_a.append(dense_a[i+1])
        
        # Wait, the logic above appends the end of segment i.
        # Segment 0 ends at Waypoint 0.
        # Segment 1 starts at Waypoint 0.
        # So we need to be careful not to duplicate or miss.
        
        # Let's restart the dense point construction more carefully.
        # Points are: Start, [Intermediates 0-1], Waypoint 0, [Intermediates 1-2], Waypoint 1, ..., Waypoint N-1, [Intermediates N-End], End.
        
        full_dense_x = [0.0]
        full_dense_y = [0.0]
        full_dense_a = [0.0]
        
        # Segment 0: Start (0,0,0) to Waypoint 0
        a1, a2 = 0.0, angles[0]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m):
            t = k / m
            full_dense_x.append(0.0 + t * (x_vals[0] - 0.0))
            full_dense_y.append(0.0 + t * (y_vals[0] - 0.0))
            full_dense_a.append(a1 + t * (a2 - a1))
        # Append Waypoint 0
        full_dense_x.append(x_vals[0])
        full_dense_y.append(y_vals[0])
        full_dense_a.append(angles[0])
        
        # Segments between Waypoints
        for i in range(N - 1):
            a1, a2 = angles[i], angles[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for k in range(1, m):
                t = k / m
                full_dense_x.append(x_vals[i] + t * (x_vals[i+1] - x_vals[i]))
                full_dense_y.append(y_vals[i] + t * (y_vals[i+1] - y_vals[i]))
                full_dense_a.append(a1 + t * (a2 - a1))
            # Append Waypoint i+1
            full_dense_x.append(x_vals[i+1])
            full_dense_y.append(y_vals[i+1])
            full_dense_a.append(angles[i+1])
        
        # Segment N: Waypoint N-1 to End (-2,0,90)
        a1, a2 = angles[N-1], 90.0
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m):
            t = k / m
            full_dense_x.append(x_vals[N-1] + t * (-2.0 - x_vals[N-1]))
            full_dense_y.append(y_vals[N-1] + t * (0.0 - y_vals[N-1]))
            full_dense_a.append(a1 + t * (a2 - a1))
        # Append End
        full_dense_x.append(-2.0)
        full_dense_y.append(0.0)
        full_dense_a.append(90.0)
        
        # Now evaluate intersection
        in_sofa = np.ones(N_pts, dtype=bool)
        
        # Convert to numpy arrays for vectorized ops if possible?
        # The loop over dense points is unavoidable for the intersection logic unless we do something fancy.
        # But we can vectorize the rotation.
        
        for idx in range(len(full_dense_x)):
            x = full_dense_x[idx]
            y = full_dense_y[idx]
            a = full_dense_a[idx]
            
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
    # Use 10 intermediate angles for more flexibility (from inspiration 1)
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 88.0])
    n_angles = len(angles)
    n_params = 2 * n_angles
    
    # Initial guess from baseline
    base_x = [-0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7]
    base_y = [0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3]
    base_a = [15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0]
    
    init_x = np.interp(angles, base_a, base_x)
    init_y = np.interp(angles, base_a, base_y)
    
    p0 = np.concatenate([init_x, init_y])
    
    # Objective function for optimizer (minimize negative area)
    def objective(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        area = evaluate_path(x_vals, y_vals, angles)
        return -area

    # Optimize
    best_area = objective(p0)
    best_params = p0.copy()
    
    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Strategy:
    # 1. Powell optimization with moderate limits.
    # 2. If time remains, Nelder-Mead polish.
    # 3. If time remains, try a second Powell run with slightly perturbed start or different parameters.
    
    try:
        res = so.minimize(objective, p0, method='Powell', 
                          options={'maxiter': 2000, 'maxfev': 2000, 'disp': False})
        
        if not time_guard():
            if res.fun < best_area:
                best_area = -res.fun
                best_params = res.x.copy()
    except Exception as e:
        pass
        
    if not time_guard():
        try:
            res2 = so.minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'maxfev': 1000, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8})
            if not time_guard():
                if res2.fun < best_area:
                    best_area = -res2.fun
                    best_params = res2.x.copy()
        except Exception:
            pass

    # If still time, try a coarse-to-fine or different restart
    if not time_guard():
        # Try restarting Powell with a small random perturbation to escape local minima
        # But be careful with time.
        try:
            # Add small noise
            noise = np.random.normal(0, 0.01, size=n_params)
            p0_restart = best_params + noise
            res3 = so.minimize(objective, p0_restart, method='Powell', 
                               options={'maxiter': 500, 'maxfev': 500, 'disp': False})
            if not time_guard():
                if res3.fun < best_area:
                    best_area = -res3.fun
                    best_params = res3.x.copy()
        except Exception:
            pass

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]
    
    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])
        
    path_arr = np.array(final_path, dtype=np.float64)
    
    return path_arr
