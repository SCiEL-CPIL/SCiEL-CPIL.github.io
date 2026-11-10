import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        if len(path) == 0:
            return np.array([P_start, P_end])
        
        full_path = np.vstack([P_start, path, P_end])
        
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        
        return np.array(dense_points)

    def score_path_fast(path):
        dense = densify_path(path)
        M = dense.shape[0]
        if M == 0:
            return 0.0
        
        alive = np.ones(n_points, dtype=bool)
        
        # Precompute cos/sin for all dense points
        a_rad = np.deg2rad(dense[:, 2])
        cos_all = np.cos(a_rad)
        sin_all = np.sin(a_rad)
        
        # Process in batches to balance memory and speed
        batch_size = max(1, min(M, 20))
        
        for batch_start in range(0, M, batch_size):
            if not np.any(alive):
                break
            
            batch_end = min(batch_start + batch_size, M)
            batch = dense[batch_start:batch_end]
            c_batch = cos_all[batch_start:batch_end]
            s_batch = sin_all[batch_start:batch_end]
            
            X_alive = X_flat[alive]
            Y_alive = Y_flat[alive]
            n_alive = len(X_alive)
            
            if n_alive == 0:
                break
            
            # For each dense point in batch, compute u,v for all alive grid points
            # Shape: (batch_len, n_alive)
            dx = X_alive[np.newaxis, :] - batch[:, 0][:, np.newaxis]
            dy = Y_alive[np.newaxis, :] - batch[:, 1][:, np.newaxis]
            
            u = dx * c_batch[:, np.newaxis] + dy * s_batch[:, np.newaxis]
            v = -dx * s_batch[:, np.newaxis] + dy * c_batch[:, np.newaxis]
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_r1 | in_r2
            
            # A point must be inside ALL dense points in this batch
            must_survive = inside.all(axis=0)
            
            alive_indices = np.where(alive)[0]
            alive[alive_indices[~must_survive]] = False
        
        count = np.count_nonzero(alive)
        return count * 0.0001

    def bspline_to_waypoints(ctrl_pts, n_samples=20):
        """Convert B-spline control points to actual waypoints."""
        n_ctrl = ctrl_pts.shape[0]
        t_ctrl = np.linspace(0, 1, n_ctrl)
        t_samples = np.linspace(0, 1, n_samples)
        
        waypoints = np.zeros((n_samples, 3))
        for d in range(3):
            cs = CubicSpline(t_ctrl, ctrl_pts[:, d])
            waypoints[:, d] = cs(t_samples)
        
        return waypoints

    def objective(ctrl_pts):
        """Objective function to minimize (negative of score)."""
        if time.time() > deadline - 30:
            return 1e10
        
        waypoints = bspline_to_waypoints(ctrl_pts, n_samples=15)
        
        # Ensure angles are monotonic
        angles = waypoints[:, 2]
        for i in range(1, len(angles)):
            if angles[i] < angles[i-1]:
                angles[i] = angles[i-1]
        
        waypoints[:, 2] = angles
        
        score = score_path_fast(waypoints)
        return -score

    def optimize_bspline(n_ctrl, n_samples=15, max_iter=50):
        """Optimize B-spline control points using SLSQP."""
        t = np.linspace(0, 1, n_ctrl)
        
        # Better initial guess: the sofa corner needs to navigate around
        # x: move from 0 to -2, with a slight inward curve
        x0 = -2.0 * t
        # y: arc that helps the sofa pass through the corner
        y0 = 0.4 * np.sin(np.pi * t)
        # angle: 0 to 90, but with a slight ease-in/ease-out
        a0 = 90.0 * t
        
        ctrl_init = np.column_stack([x0, y0, a0])
        
        # Add some randomness
        ctrl_init[:, 1] += np.random.normal(0, 0.08, n_ctrl)
        ctrl_init[:, 1] = np.clip(ctrl_init[:, 1], -0.3, 1.2)
        
        # Bounds for control points
        bounds = [
            (-3.0, 1.0),  # x range
            (-1.0, 2.0),  # y range
            (0.0, 90.0)   # angle range
        ] * n_ctrl
        
        # Constraints: angle monotonicity
        def angle_mono_constraint(x):
            ctrl = x.reshape(-1, 3)
            angles = ctrl[:, 2]
            return np.diff(angles)
        
        constraints = [
            {'type': 'ineq', 'fun': angle_mono_constraint}
        ]
        
        try:
            result = minimize(
                objective,
                ctrl_init.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-5}
            )
            
            if result.success or result.fun < 1e9:
                best_ctrl = result.x.reshape(-1, 3)
                # Ensure monotonic angles
                for i in range(1, best_ctrl.shape[0]):
                    if best_ctrl[i, 2] < best_ctrl[i-1, 2]:
                        best_ctrl[i, 2] = best_ctrl[i-1, 2]
                return best_ctrl
        except Exception:
            pass
        
        return ctrl_init

    best_path = None
    best_score = 0.0
    
    # Try different numbers of control points
    for n_ctrl in [4, 5, 6, 7]:
        if time.time() > deadline - 60:
            break
        
        # Multiple random restarts
        for restart in range(4):
            if time.time() > deadline - 40:
                break
            
            np.random.seed(42 + restart * 100 + n_ctrl)
            
            ctrl = optimize_bspline(n_ctrl, max_iter=40)
            
            # Convert to waypoints and evaluate
            waypoints = bspline_to_waypoints(ctrl, n_samples=14)
            
            # Ensure monotonic angles
            angles = waypoints[:, 2]
            for i in range(1, len(angles)):
                if angles[i] < angles[i-1]:
                    angles[i] = angles[i-1]
            waypoints[:, 2] = angles
            
            s = score_path_fast(waypoints)
            if s > best_score:
                best_score = s
                best_path = waypoints.copy()
    
    # If B-spline didn't find anything good, fall back to direct waypoint optimization
    if best_path is None or best_score < 0.01:
        def make_initial_path(n_wp):
            angles = np.linspace(0, 90, n_wp + 2)[1:-1]
            angles += np.random.normal(0, 1.5, n_wp)
            angles = np.maximum.accumulate(angles)
            angles = np.clip(angles, 1, 89)
            
            t = np.linspace(0, 1, n_wp)
            xs_init = -2.0 * t + np.random.normal(0, 0.15, n_wp)
            ys_init = 0.4 * np.sin(np.pi * t) + np.random.normal(0, 0.15, n_wp)
            
            return np.column_stack([xs_init, ys_init, angles])
        
        np.random.seed(42)
        
        for n_wp in [5, 7, 9]:
            if time.time() > deadline - 30:
                break
            
            path = make_initial_path(n_wp)
            s = score_path_fast(path)
            if s > best_score:
                best_score = s
                best_path = path.copy()
    
    # Final hill climbing refinement with adaptive step size
    if best_path is not None and time.time() < deadline - 20:
        current_path = best_path.copy()
        current_score = score_path_fast(current_path)
        
        sigma = 0.12
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            n_coords = current_path.shape[0] * 3
            
            # Try perturbing multiple coordinates
            n_tries = min(8, max(3, (int(deadline - 10) - int(time.time())) // 5))
            
            for _ in range(n_tries):
                if time.time() > deadline - 10:
                    break
                
                # Pick 1-2 coordinates to perturb
                n_perturb = 1 if np.random.random() < 0.7 else 2
                coord_indices = np.random.choice(n_coords, size=n_perturb, replace=False)
                
                best_delta = None
                best_new_score = current_score
                
                for coord_idx in coord_indices:
                    row = coord_idx // 3
                    col = coord_idx % 3
                    
                    for delta in [sigma, -sigma, sigma*0.5, -sigma*0.5, sigma*0.25, -sigma*0.25]:
                        new_path = current_path.copy()
                        new_path[row, col] += delta
                        
                        if col == 2:
                            if row > 0:
                                new_path[row, 2] = max(new_path[row, 2], new_path[row-1, 2])
                            if row < current_path.shape[0] - 1:
                                new_path[row, 2] = min(new_path[row, 2], new_path[row+1, 2])
                        
                        new_score = score_path_fast(new_path)
                        
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_delta = (row, col, delta)
                
                if best_delta is not None and best_new_score > current_score:
                    current_path[best_delta[0], best_delta[1]] += best_delta[2]
                    if best_delta[1] == 2:
                        r = best_delta[0]
                        if r > 0:
                            current_path[r, 2] = max(current_path[r, 2], current_path[r-1, 2])
                        if r < current_path.shape[0] - 1:
                            current_path[r, 2] = min(current_path[r, 2], current_path[r+1, 2])
                    current_score = best_new_score
                    improved = True
                    break
            
            if improved:
                no_improve = 0
                sigma *= 0.92
            else:
                no_improve += 1
                sigma *= 0.95
        
        if current_score > best_score:
            best_score = current_score
            best_path = current_path.copy()
    
    if best_path is None:
        best_path = np.array([[-0.5, 0.3, 22.5], [-1.0, 0.5, 45.0], [-1.5, 0.3, 67.5]])
    
    return best_path
