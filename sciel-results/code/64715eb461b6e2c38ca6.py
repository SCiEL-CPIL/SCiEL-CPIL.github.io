import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)

    start_time = time.time()
    time_limit = 580.0  # Safety margin before 630s kill

    def _make_densified(path):
        """Build the full densified path: start + path + end, with linear interpolation."""
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                dens.append((
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    p1[2] + (p2[2] - p1[2]) * t,
                ))
        dens.append(full[-1])
        return dens

    def evaluate_path_fast(path):
        """Fast evaluation using precomputed grid and vectorized rotation."""
        dens = _make_densified(path)
        n_dens = len(dens)
        # Precompute all rotations
        angles_rad = np.array([np.deg2rad(d[2]) for d in dens])
        cos_arr = np.cos(angles_rad)
        sin_arr = np.sin(angles_rad)
        cx_arr = np.array([d[0] for d in dens])
        cy_arr = np.array([d[1] for d in dens])
        
        # For each placement, compute mask and take intersection
        # Process in batches to limit memory
        mask = np.ones(n_points, dtype=bool)
        for i in range(n_dens):
            cos_a = cos_arr[i]
            sin_a = sin_arr[i]
            dx = x_flat - cx_arr[i]
            dy = y_flat - cy_arr[i]
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask &= (in_arm1 | in_arm2)
            if not np.any(mask):
                return 0.0
        return float(np.sum(mask)) * 0.0001

    def evaluate_path(path):
        """Full evaluation: returns area score."""
        return evaluate_path_fast(path)

    # Multiple initial paths to try
    initial_paths = [
        [
            (-0.1, 0.1, 15.0),
            (-0.2, 0.2, 20.0),
            (-0.3, 0.3, 34.0),
            (-0.3, 0.3, 40.0),
            (-0.4, 0.3, 50.0),
            (-0.5, 0.2, 55.0),
            (-0.6, 0.2, 75.0),
            (-0.7, 0.3, 86.0),
        ],
        [
            (-0.05, 0.05, 10.0),
            (-0.15, 0.15, 25.0),
            (-0.25, 0.25, 40.0),
            (-0.35, 0.25, 55.0),
            (-0.45, 0.2, 70.0),
            (-0.55, 0.25, 82.0),
            (-0.65, 0.3, 88.0),
        ],
        [
            (-0.1, 0.0, 12.0),
            (-0.2, 0.1, 28.0),
            (-0.3, 0.2, 42.0),
            (-0.35, 0.2, 55.0),
            (-0.45, 0.15, 68.0),
            (-0.55, 0.2, 80.0),
            (-0.65, 0.25, 88.0),
            (-0.72, 0.28, 90.0),
        ],
    ]

    best_score = 0.0
    best_path = None
    current_path = None

    # Evaluate all initial paths and pick the best
    for ip in initial_paths:
        s = evaluate_path(ip)
        if s > best_score:
            best_score = s
            best_path = ip[:]
            current_path = ip[:]

    # Phase 1: Random local search with adaptive step sizes and restarts
    np.random.seed(42)
    epsilon_xy = 0.12
    epsilon_a = 2.5
    no_improve = 0
    max_no_improve = 25
    phase1_time_limit = 300.0  # Spend ~300s on random search, leave more for spline
    restart_count = 0
    max_restarts = 3

    while time.time() - start_time < min(time_limit, phase1_time_limit):
        idx = np.random.randint(len(current_path))
        p = current_path[idx]

        new_x = p[0] + np.random.normal(0, epsilon_xy)
        new_y = p[1] + np.random.normal(0, epsilon_xy)
        new_a = p[2] + np.random.normal(0, epsilon_a)
        new_a = np.clip(new_a, 0.0, 90.0)

        new_p = (new_x, new_y, new_a)
        test_path = current_path[:]
        test_path[idx] = new_p

        score = evaluate_path(test_path)

        if score > best_score:
            best_score = score
            best_path = test_path
            current_path = test_path
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                epsilon_xy *= 0.8
                epsilon_a *= 0.8
                no_improve = 0
                if epsilon_xy < 0.008 and epsilon_a < 0.4:
                    # Restart from best with fresh noise
                    current_path = best_path[:]
                    epsilon_xy = 0.25
                    epsilon_a = 4.0
                    restart_count += 1
                    if restart_count >= max_restarts:
                        # Try a different initial path
                        for ip in initial_paths:
                            if ip is not best_path:
                                current_path = ip[:]
                                break
                        else:
                            current_path = best_path[:]
                        restart_count = 0

    # Phase 1.5: Spline-based optimization using Nelder-Mead
    # Represent the path as a cubic B-spline parameterized by angle (0 to 90)
    # The spline gives (x(t), y(t), a(t)) where a(t) = t * 90
    # We optimize the control points of x(t) and y(t)
    
    def spline_to_path(ctrl_x, ctrl_y, n_knots=12):
        """Convert B-spline control points to a discrete path.
        ctrl_x, ctrl_y: 1D arrays of length n_knots
        Returns list of (x, y, a) tuples with a from 0 to 90.
        """
        # Parameter t in [0, 1], a = 90*t
        # Use simple cubic interpolation via scipy
        from scipy.interpolate import CubicSpline
        t_ctrl = np.linspace(0, 1, n_knots)
        try:
            cs_x = CubicSpline(t_ctrl, ctrl_x, bc_type='natural')
            cs_y = CubicSpline(t_ctrl, ctrl_y, bc_type='natural')
        except Exception:
            # Fallback to linear
            t_eval = np.linspace(0, 1, 20)
            x_eval = np.interp(t_eval, t_ctrl, ctrl_x)
            y_eval = np.interp(t_eval, t_ctrl, ctrl_y)
        else:
            t_eval = np.linspace(0, 1, 20)
            x_eval = cs_x(t_eval)
            y_eval = cs_y(t_eval)
        
        path = []
        for i in range(len(t_eval)):
            a_val = 90.0 * t_eval[i]
            if 0.5 <= a_val <= 89.5:  # Skip start and end (they're fixed)
                path.append((float(x_eval[i]), float(y_eval[i]), float(a_val)))
        return path

    def nm_objective(params):
        """Nelder-Mead objective: minimize negative score (maximize area).
        params: flat array of [ctrl_x..., ctrl_y...]
        """
        if time.time() > time_limit - 60:
            return 1.0  # Penalize late evaluations
        n_knots = len(params) // 2
        ctrl_x = params[:n_knots]
        ctrl_y = params[n_knots:]
        
        # Ensure smoothness: penalize large jumps
        if n_knots > 1:
            dx = np.diff(ctrl_x)
            dy = np.diff(ctrl_y)
            smoothness_penalty = 0.01 * (np.sum(dx**2) + np.sum(dy**2))
        else:
            smoothness_penalty = 0.0
        
        path = spline_to_path(ctrl_x, ctrl_y, n_knots)
        if not path:
            return 1.0 + smoothness_penalty
        
        score = evaluate_path(path)
        return -score + smoothness_penalty

    # Try spline optimization with different numbers of knots
    spline_deadline = time_limit - 120.0  # Leave 120s for final polish
    
    for n_knots in [6, 8, 10]:
        if time.time() > spline_deadline:
            break
        
        # Initialize control points from best_path
        # Map best_path angles to [0,1] and interpolate to n_knots points
        bp_angles = np.array([p[2] for p in best_path])
        bp_x = np.array([p[0] for p in best_path])
        bp_y = np.array([p[1] for p in best_path])
        
        t_ctrl = np.linspace(0, 1, n_knots)
        a_ctrl = t_ctrl * 90.0
        
        # Interpolate x and y at control point angles
        # Extend with endpoints
        all_a = np.concatenate([[0.0], bp_angles, [90.0]])
        all_x = np.concatenate([[0.0], bp_x, [-2.0]])
        all_y = np.concatenate([[0.0], bp_y, [0.0]])
        
        ctrl_x_init = np.interp(a_ctrl, all_a, all_x)
        ctrl_y_init = np.interp(a_ctrl, all_a, all_y)
        
        x0 = np.concatenate([ctrl_x_init, ctrl_y_init])
        
        try:
            result = minimize(
                nm_objective, x0, method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 0.005,
                    'fatol': 1e-5,
                    'adaptive': True,
                }
            )
            
            if result.success or result.fun < 0:
                ctrl_x_opt = result.x[:n_knots]
                ctrl_y_opt = result.x[n_knots:]
                spline_path = spline_to_path(ctrl_x_opt, ctrl_y_opt, n_knots)
                
                if spline_path:
                    s = evaluate_path(spline_path)
                    if s > best_score:
                        best_score = s
                        best_path = spline_path
        except Exception:
            pass

    # Phase 2: Coordinate-descent polish with multiple scales
    polish_deadline = time_limit - 30.0  # Leave 30s buffer for final polish

    # Multi-scale polish: start coarse, get finer
    scales = [
        (0.04, 0.8),
        (0.02, 0.4),
        (0.01, 0.2),
        (0.005, 0.1),
        (0.002, 0.05),
        (0.001, 0.02),
        (0.0005, 0.01),
    ]

    for polish_xy, polish_a in scales:
        if time.time() > polish_deadline:
            break
        # Multiple passes at each scale
        for pass_iter in range(3):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(len(best_path)):
                if time.time() > polish_deadline:
                    break
                p = best_path[idx]
                best_local = best_score
                best_local_p = p

                # Try perturbations in x
                for dx in (-polish_xy, -polish_xy * 0.5, polish_xy * 0.5, polish_xy):
                    test_p = (p[0] + dx, p[1], p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                # Try perturbations in y
                for dy in (-polish_xy, -polish_xy * 0.5, polish_xy * 0.5, polish_xy):
                    test_p = (p[0], p[1] + dy, p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                # Try perturbations in a
                for da in (-polish_a, -polish_a * 0.5, polish_a * 0.5, polish_a):
                    new_a = np.clip(p[2] + da, 0.0, 90.0)
                    test_p = (p[0], p[1], new_a)
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                if best_local > best_score:
                    best_path[idx] = best_local_p
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 3: Final high-precision polish with very small steps
    if time.time() < polish_deadline:
        fine_xy = 0.0002
        fine_a = 0.005
        for pass_iter in range(5):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(len(best_path)):
                if time.time() > polish_deadline:
                    break
                p = best_path[idx]
                best_local = best_score
                best_local_p = p

                for dx in (-fine_xy, fine_xy):
                    test_p = (p[0] + dx, p[1], p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                for dy in (-fine_xy, fine_xy):
                    test_p = (p[0], p[1] + dy, p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                for da in (-fine_a, fine_a):
                    new_a = np.clip(p[2] + da, 0.0, 90.0)
                    test_p = (p[0], p[1], new_a)
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p

                if best_local > best_score:
                    best_path[idx] = best_local_p
                    best_score = best_local
                    improved = True

            if not improved:
                break

    return np.array(best_path)
