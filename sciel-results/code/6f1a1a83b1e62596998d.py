# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Uses parametric optimization: path is x(theta), y(theta) as smooth functions
    of angle. Optimize coefficients to maximize surviving sofa area.
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
        Generate path from polynomial coefficients.
        coeffs = [x0, x1, x2, x3, y0, y1, y2, y3]
        x(theta) = x0 + x1*t + x2*t^2 + x3*t^3, where t = theta/90
        y(theta) = y0 + y1*t + y2*t^2 + y3*t^3
        Constraints: x(0)=0, y(0)=0, x(1)=-2, y(1)=0
        So: x0=0, x1+x2+x3=-2, y0=0, y1+y2+y3=0
        Free params: x1, x2, y1, y2 (x3 and y3 determined)
        """
        x1, x2, y1, y2 = coeffs
        x3 = -2.0 - x1 - x2
        y3 = -y1 - y2

        theta = np.linspace(5, 85, n_theta)
        t = theta / 90.0

        xs = x1 * t + x2 * t**2 + x3 * t**3
        ys = y1 * t + y2 * t**2 + y3 * t**3

        path = np.column_stack((xs, ys, theta))
        return path

    def objective(coeffs):
        """Negative area to minimize."""
        if time.time() > deadline - 5:
            return 0.0
        path = get_path_from_coeffs(coeffs)
        score = evaluate_path_fast(path)
        return -score

    # Phase 1: Try multiple initializations with different strategies
    best_path = None
    best_score = -1.0

    # Known good starting point
    initial_coeffs_list = [
        # [x1, x2, y1, y2]
        [-2.0, 0.0, 0.3, 0.0],    # linear x, small y bump
        [-1.5, -0.5, 0.2, 0.1],
        [-1.0, -1.0, 0.4, -0.2],
        [-2.0, 0.0, 0.0, 0.0],    # pure linear
        [-1.8, -0.2, 0.25, -0.1],
        [-1.2, -0.8, 0.35, 0.05],
        [-2.2, 0.2, 0.15, 0.1],
        [-1.6, -0.4, 0.45, -0.15],
    ]

    rng = np.random.default_rng(42)

    for init in initial_coeffs_list:
        if time.time() > deadline - 30:
            break

        # Add small random perturbation for diversity
        perturbed = np.array(init) + rng.uniform(-0.1, 0.1, 4)

        # Use Nelder-Mead for local optimization
        try:
            result = opt.minimize(
                objective,
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
        except Exception:
            result = None

        if result is not None and result.fun < -best_score:
            best_score = -result.fun
            best_path = get_path_from_coeffs(result.x)
        elif best_path is None:
            best_path = get_path_from_coeffs(perturbed)
            best_score = evaluate_path_fast(best_path)

    # Phase 2: Try with more polynomial terms (quintic)
    def get_path_from_coeffs5(coeffs, n_theta=100):
        """Quintic: x(t) = x1*t + x2*t^2 + x3*t^3 + x4*t^4 + x5*t^5, x(0)=0, x(1)=-2"""
        x1, x2, x3, x4, y1, y2, y3, y4 = coeffs
        x5 = -2.0 - x1 - x2 - x3 - x4
        y5 = -y1 - y2 - y3 - y4

        theta = np.linspace(5, 85, n_theta)
        t = theta / 90.0

        xs = x1*t + x2*t**2 + x3*t**3 + x4*t**4 + x5*t**5
        ys = y1*t + y2*t**2 + y3*t**3 + y4*t**4 + y5*t**5

        path = np.column_stack((xs, ys, theta))
        return path

    def objective5(coeffs):
        if time.time() > deadline - 5:
            return 0.0
        path = get_path_from_coeffs5(coeffs)
        score = evaluate_path_fast(path)
        return -score

    if time.time() < deadline - 40:
        init5_list = [
            [-2.0, 0, 0, 0, 0.3, 0, 0, 0],
            [-1.5, -0.5, 0, 0, 0.2, 0.1, 0, 0],
            [-1.0, -1.0, 0.5, 0, 0.4, -0.2, 0, 0],
            [-2.0, 0, 0, 0, 0.0, 0, 0, 0],
        ]
        for init in init5_list:
            if time.time() > deadline - 20:
                break
            perturbed = np.array(init) + rng.uniform(-0.05, 0.05, 8)
            try:
                result = opt.minimize(
                    objective5,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
                )
            except Exception:
                result = None
            if result is not None and result.fun < -best_score:
                best_score = -result.fun
                best_path = get_path_from_coeffs5(result.x)

    # Phase 3: Refine best path with local waypoint perturbation
    if best_path is not None and time.time() < deadline - 15:
        # Convert to waypoint representation and do local search
        for it in range(100):
            if time.time() > deadline - 10:
                break
            idx = rng.integers(0, len(best_path))
            new_path = best_path.copy()
            new_path[idx, 0] += rng.uniform(-0.05, 0.05)
            new_path[idx, 1] += rng.uniform(-0.05, 0.05)
            new_path[idx, 2] = np.clip(new_path[idx, 2] + rng.uniform(-0.3, 0.3), 1, 89)
            score = evaluate_path_fast(new_path)
            if score > best_score:
                best_score = score
                best_path = new_path

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
