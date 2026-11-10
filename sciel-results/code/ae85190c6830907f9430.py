# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Combines search-based initialization with parametric optimization.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s kill

    # Setup grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]

    wx = P[:, 0].copy()
    wy = P[:, 1].copy()

    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path_array):
        """Densify path so angle steps are <= 1 degree."""
        if path_array is None or len(path_array) == 0:
            return np.array([start_state, end_state])

        full = np.vstack([start_state, path_array, end_state])

        dense = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            a1 = p1[2]
            a2 = p2[2]
            da = a2 - a1
            steps = max(1, int(np.ceil(abs(da))))

            if steps == 1:
                dense.append(p2)
            else:
                dx = (p2[0] - p1[0]) / steps
                dy = (p2[1] - p1[1]) / steps
                da_step = da / steps
                for j in range(1, steps):
                    dense.append([p1[0] + j*dx, p1[1] + j*dy, a1 + j*da_step])
                dense.append(p2)

        return np.array(dense)

    def evaluate_path_fast(path_array):
        """Evaluate area of sofa surviving all placements."""
        dense = densify_path(path_array)
        
        a_rad = np.deg2rad(dense[:, 2])
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        cx = dense[:, 0]
        cy = dense[:, 1]

        mask = np.ones(N_points, dtype=bool)

        for idx in range(len(dense)):
            c = cos_a[idx]
            s = sin_a[idx]
            x = cx[idx]
            y = cy[idx]

            dx = wx - x
            dy = wy - y

            u = dx * c + dy * s
            v = -dx * s + dy * c

            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_corridor = in_r1 | in_r2

            mask &= in_corridor

            if not mask.any():
                return 0.0

        return float(np.sum(mask)) * 0.0001

    def get_path_from_coeffs(coeffs, n_theta=100):
        """
        Generate path from cubic polynomial coefficients.
        coeffs = [A, B, E, F]
        x(t) = A*t^3 + B*t^2 + (-2 - A - B)*t
        y(t) = E*t^3 + F*t^2 + (-E - F)*t
        t = theta/90
        """
        A, B, E, F = coeffs
        C = -2.0 - A - B
        G = -E - F

        theta = np.linspace(5, 85, n_theta)
        t = theta / 90.0

        xs = A * t**3 + B * t**2 + C * t
        ys = E * t**3 + F * t**2 + G * t

        path = np.column_stack((xs, ys, theta))
        return path

    def objective(coeffs):
        """Negative area to minimize."""
        if time.time() > deadline - 5:
            return 0.0
        path = get_path_from_coeffs(coeffs)
        score = evaluate_path_fast(path)
        return -score

    def fit_cubic_to_path(path):
        """
        Fit a cubic polynomial to a given path (waypoints).
        Returns coeffs [A, B, E, F].
        """
        if len(path) < 4:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Sort by angle
        path = path[path[:, 2].argsort()]
        
        t = path[:, 2] / 90.0
        xs = path[:, 0]
        ys = path[:, 1]
        
        # Fit x(t) = A*t^3 + B*t^2 + C*t + D
        # We enforce x(0)=0, x(1)=-2.
        # So we fit x(t) - (-2*t) = A*t^3 + B*t^2 + C'*t where C' is free?
        # Actually, let's just do a least squares fit on the free coefficients.
        # x(t) = A*t^3 + B*t^2 + C*t. Constraint: A+B+C = -2.
        # Let x(t) = A*t^3 + B*t^2 + (-2 - A - B)*t = A*(t^3 - t) + B*(t^2 - t) - 2*t.
        # Let u(t) = t^3 - t, v(t) = t^2 - t.
        # x(t) + 2*t = A*u(t) + B*v(t).
        
        u = t**3 - t
        v = t**2 - t
        
        # Fit x(t) + 2*t = A*u + B*v
        # Design matrix
        M_x = np.column_stack((u, v))
        target_x = xs + 2.0 * t
        
        # Fit y(t) = E*t^3 + F*t^2 + G*t. Constraint: E+F+G=0.
        # y(t) = E*(t^3 - t) + F*(t^2 - t).
        # target_y = ys
        M_y = np.column_stack((u, v))
        target_y = ys
        
        try:
            coeff_x, _, _, _ = np.linalg.lstsq(M_x, target_x, rcond=None)
            A, B = coeff_x
            coeff_y, _, _, _ = np.linalg.lstsq(M_y, target_y, rcond=None)
            E, F = coeff_y
        except:
            return np.array([0.0, 0.0, 0.0, 0.0])
            
        return np.array([A, B, E, F])

    def generate_random_path(n_waypoints, rng):
        """Generate a random path with n_waypoints intermediate points."""
        angles = rng.uniform(2, 88, n_waypoints)
        angles = np.sort(angles)
        
        # x should go from ~0 to ~-2
        # y should stay in a reasonable range
        xs = rng.uniform(-2.5, 0.5, n_waypoints)
        ys = rng.uniform(-0.5, 0.5, n_waypoints)
        
        # Make x roughly monotonic decreasing
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        angles = angles[order]
        
        # Ensure x trends from 0 to -2
        t = angles / 90.0
        x_linear = -2.0 * t
        xs = 0.7 * xs + 0.3 * x_linear
        
        path = np.column_stack((xs, ys, angles))
        return path

    # Phase 1: Generate diverse random candidates and fit cubics
    rng = np.random.default_rng(42)
    
    best_path = None
    best_score = -1.0
    
    initial_coeffs_list = []
    
    # Generate random paths, evaluate, keep best, fit cubics
    n_random_trials = 200
    for i in range(n_random_trials):
        if time.time() > deadline - 100:
            break
            
        # Vary number of waypoints
        n_wp = rng.integers(5, 15)
        path = generate_random_path(n_wp, rng)
        score = evaluate_path_fast(path)
        
        if score > 0.1: # Only fit if it's not terrible
            coeffs = fit_cubic_to_path(path)
            # Add small perturbation
            coeffs += rng.uniform(-0.1, 0.1, 4)
            initial_coeffs_list.append(coeffs)
            
            if score > best_score:
                best_score = score
                best_path = path.copy()

    # Add some hand-crafted initializations
    initial_coeffs_list.append(np.array([-2.0, 0.0, 0.3, 0.0]))
    initial_coeffs_list.append(np.array([-1.5, -0.5, 0.2, 0.1]))
    initial_coeffs_list.append(np.array([-1.0, -1.0, 0.4, -0.2]))
    initial_coeffs_list.append(np.array([-2.0, 0.0, 0.0, 0.0]))

    # Phase 2: Parametric Optimization with Nelder-Mead
    for coeffs in initial_coeffs_list:
        if time.time() > deadline - 30:
            break
            
        try:
            result = opt.minimize(
                objective,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
        except Exception:
            continue

        if result.fun < -best_score:
            best_score = -result.fun
            best_path = get_path_from_coeffs(result.x)

    # Phase 3: Polish with Waypoint Local Search
    if best_path is not None and time.time() < deadline - 10:
        # Convert best path to waypoints for local search
        # Densify it first to have more points to perturb
        dense_path = densify_path(best_path)
        # We can perturb the original waypoints of best_path if it was from get_path_from_coeffs
        # But best_path from get_path_from_coeffs has 100 points.
        # Let's just use the 100 points.
        
        current_path = best_path.copy()
        current_score = best_score
        
        for it in range(500):
            if time.time() > deadline - 5:
                break
                
            idx = rng.integers(0, len(current_path))
            new_path = current_path.copy()
            new_path[idx, 0] += rng.uniform(-0.02, 0.02)
            new_path[idx, 1] += rng.uniform(-0.02, 0.02)
            new_path[idx, 2] = np.clip(new_path[idx, 2] + rng.uniform(-0.1, 0.1), 1, 89)
            
            score = evaluate_path_fast(new_path)
            if score > current_score:
                current_score = score
                current_path = new_path
                best_score = score
                best_path = current_path

    # Final fallback
    if best_path is None or not np.all(np.isfinite(best_path)):
        best_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 25.0],
            [-0.3, 0.3, 35.0],
            [-0.4, 0.3, 45.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 65.0],
            [-0.7, 0.2, 75.0],
            [-0.8, 0.1, 85.0],
        ], dtype=np.float64)

    return best_path

# EVOLVE-BLOCK-END
