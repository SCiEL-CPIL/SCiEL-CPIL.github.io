import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0

    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Sofa points definition
    # Full grid
    i_vals_full = np.arange(-3.0, 1.5, 0.01)
    j_vals_full = np.arange(-1.5, 1.5, 0.01)
    P_x_full, P_y_full = np.meshgrid(i_vals_full, j_vals_full, indexing='xy')
    P_x_full = P_x_full.flatten()
    P_y_full = P_y_full.flatten()
    N_pts_full = len(P_x_full)

    # Coarse grid (3x coarser)
    i_vals_coarse = np.arange(-3.0, 1.5, 0.03)
    j_vals_coarse = np.arange(-1.5, 1.5, 0.03)
    P_x_coarse, P_y_coarse = np.meshgrid(i_vals_coarse, j_vals_coarse, indexing='xy')
    P_x_coarse = P_x_coarse.flatten()
    P_y_coarse = P_y_coarse.flatten()
    N_pts_coarse = len(P_x_coarse)

    def build_dense_structure(angles):
        N = len(angles)
        n_anchors = N + 2
        dense_start_idx = []
        dense_t = []

        # Segment 0: Start (0) to Waypoint 0 (1)
        a1, a2 = 0.0, angles[0]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        dense_start_idx.append(0)
        dense_t.append(0.0)
        if m > 1:
            for k in range(1, m):
                t = k / m
                dense_start_idx.append(0)
                dense_t.append(t)
        dense_start_idx.append(0) # Start anchor for the end point
        dense_t.append(1.0)

        # Segments between waypoints
        for i in range(N - 1):
            a1, a2 = angles[i], angles[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    dense_start_idx.append(i + 1)
                    dense_t.append(t)
            dense_start_idx.append(i + 1) # Start anchor for the end point
            dense_t.append(1.0)

        # Segment N: Waypoint N-1 (N) to End (N+1)
        a1, a2 = angles[N-1], 90.0
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m > 1:
            for k in range(1, m):
                t = k / m
                dense_start_idx.append(N)
                dense_t.append(t)
        dense_start_idx.append(N) # Start anchor for the end point
        dense_t.append(1.0)

        n_dense = len(dense_start_idx)
        # Build weight matrix
        W = np.zeros((n_dense, n_anchors))
        for i in range(n_dense):
            a_idx = dense_start_idx[i]
            t = dense_t[i]
            W[i, a_idx] = 1.0 - t
            W[i, a_idx + 1] = t

        # Compute angles for dense points
        anchor_angles = np.concatenate([[0.0], angles, [90.0]])
        dense_a = W @ anchor_angles

        return W, dense_a

    def make_evaluator(angles, P_x, P_y, N_pts):
        W, dense_a = build_dense_structure(angles)
        n_dense, n_anchors = W.shape
        cos_a = np.cos(np.deg2rad(dense_a))
        sin_a = np.sin(np.deg2rad(dense_a))

        def evaluate(x_vals, y_vals):
            # Anchor positions
            anchor_x = np.zeros(n_anchors)
            anchor_y = np.zeros(n_anchors)
            if n_anchors > 1:
                anchor_x[1:-1] = x_vals
                anchor_y[1:-1] = y_vals
            if n_anchors > 0:
                anchor_x[-1] = -2.0
                anchor_y[-1] = 0.0

            dense_x = W @ anchor_x
            dense_y = W @ anchor_y

            in_sofa = np.ones(N_pts, dtype=bool)

            for k in range(n_dense):
                dx = P_x - dense_x[k]
                dy = P_y - dense_y[k]

                u = dx * cos_a[k] + dy * sin_a[k]
                v = -dx * sin_a[k] + dy * cos_a[k]

                cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                mask = cond1 | cond2

                in_sofa &= mask

                if not np.any(in_sofa):
                    return 0.0

            return np.sum(in_sofa) * 0.0001

        return evaluate

    # Initial configuration
    # Let's start with 10 waypoints for better resolution
    n_angles = 10
    angles = np.linspace(5.0, 85.0, n_angles)
    
    # Initial guess
    init_x = np.linspace(0.0, -2.0, n_angles + 2)[1:-1]
    init_y = np.zeros(n_angles)
    
    p0 = np.concatenate([init_x, init_y])

    # Evaluator for coarse grid
    evaluator_coarse = make_evaluator(angles, P_x_coarse, P_y_coarse, N_pts_coarse)
    
    def objective_coarse(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        return -evaluator_coarse(x_vals, y_vals)

    best_params = p0.copy()
    best_area_coarse = -objective_coarse(p0)

    # Optimization Phase 1: Coarse Grid
    # Use Nelder-Mead for speed
    try:
        res = so.minimize(objective_coarse, p0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'maxfev': 5000, 'disp': False, 'xatol': 1e-4, 'fatol': 1e-6})
        if not time_guard():
            if res.fun < -best_area_coarse:
                best_area_coarse = -res.fun
                best_params = res.x
    except Exception:
        pass

    if not time_guard():
        # Try Powell on coarse grid from best NM result
        try:
            res2 = so.minimize(objective_coarse, best_params, method='Powell',
                               options={'maxiter': 500, 'maxfev': 2000, 'disp': False})
            if not time_guard():
                if res2.fun < -best_area_coarse:
                    best_area_coarse = -res2.fun
                    best_params = res2.x
        except Exception:
            pass

    # Optimization Phase 2: Full Grid Polish
    if not time_guard():
        evaluator_full = make_evaluator(angles, P_x_full, P_y_full, N_pts_full)
        
        def objective_full(params):
            x_vals = params[:n_angles]
            y_vals = params[n_angles:]
            return -evaluator_full(x_vals, y_vals)
            
        best_params_full = best_params.copy()
        best_area_full = -objective_full(best_params_full)
        
        # Short polish with Nelder-Mead
        try:
            res3 = so.minimize(objective_full, best_params_full, method='Nelder-Mead',
                               options={'maxiter': 200, 'maxfev': 500, 'disp': False, 'xatol': 1e-5, 'fatol': 1e-7})
            if not time_guard():
                if res3.fun < -best_area_full:
                    best_area_full = -res3.fun
                    best_params_full = res3.x
        except Exception:
            pass

        # Use the best full-grid result
        best_params = best_params_full

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]

    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])

    path_arr = np.array(final_path, dtype=np.float64)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(path_arr)):
        return np.array([[0.0, 0.0, 0.0]])

    return path_arr
