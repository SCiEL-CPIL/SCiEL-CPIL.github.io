import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0  # Hard limit
    safety_margin = 45.0  # Stop 45s before limit

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

    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Function to evaluate the area for a given path (list of [x, y, angle])
    def evaluate_path(path):
        # Densify the path as the evaluator does
        # Prepend (0,0,0) and append (-2,0,90)
        full_path = [(0.0, 0.0, 0.0)] + path + [(-2.0, 0.0, 90.0)]

        densified = []
        for i in range(len(full_path) - 1):
            x1, y1, a1 = full_path[i]
            x2, y2, a2 = full_path[i + 1]

            # m = max(1, ceil(|a2 - a1|))
            da = abs(a2 - a1)
            m = max(1, int(np.ceil(da)))

            # Linear interpolation in (x, y, a)
            # "in m equal steps" -> t_k = k/m for k=0..m
            # Endpoints are already in full_path, so add k=1..m-1
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    a = a1 + t * (a2 - a1)
                    densified.append((x, y, a))

        # Append the last point
        densified.append(full_path[-1])
        # Prepend the first point
        densified = [full_path[0]] + densified

        # Now check intersection
        # in_sofa is True for all points initially
        in_sofa = np.ones(N_pts, dtype=bool)

        # Process each placement
        for x, y, a in densified:
            theta = np.deg2rad(a)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # R(-a) (w - (x,y))
            dx = P_x - x
            dy = P_y - y

            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t

            # Base corridor:
            # {0 <= u <= 1, -3 <= v <= 1} OR {-3 <= u <= 1, 0 <= v <= 1}
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = cond1 | cond2

            in_sofa &= mask

            # Early exit if no points left
            if not np.any(in_sofa):
                return 0.0

        return np.sum(in_sofa) * 0.0001

    # Parameterization
    # We optimize the intermediate points.
    # Let's use 8 intermediate angles for better resolution.
    angles = np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 80.0, 86.0])
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

        # Construct path
        path = []
        for i in range(n_angles):
            path.append([x_vals[i], y_vals[i], angles[i]])

        area = evaluate_path(path)
        return -area

    # Optimize
    best_area = objective(p0)
    best_params = p0.copy()

    # Use Powell method first, robust for non-smooth
    try:
        res = so.minimize(objective, p0, method='Powell',
                          options={'maxiter': 2000, 'maxfev': 2000, 'disp': False})

        if not time_guard():
            if res.fun < best_area:
                best_area = -res.fun
                best_params = res.x.copy()
    except Exception as e:
        pass

    # If time remains, try Nelder-Mead with the best params so far
    if not time_guard():
        try:
            res2 = so.minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'maxfev': 1000, 'disp': False,
                                        'xatol': 1e-6, 'fatol': 1e-8})
            if not time_guard():
                if res2.fun < best_area:
                    best_area = -res2.fun
                    best_params = res2.x.copy()
        except Exception:
            pass

    # If time remains, do a local refinement with smaller steps
    if not time_guard():
        # Try a few random perturbations around the best solution
        n_perturbations = 5
        for _ in range(n_perturbations):
            if time_guard():
                break
            noise = np.random.normal(0, 0.05, n_params)
            perturbed = best_params + noise
            try:
                res3 = so.minimize(objective, perturbed, method='Nelder-Mead',
                                   options={'maxiter': 200, 'maxfev': 200, 'disp': False,
                                            'xatol': 1e-7, 'fatol': 1e-9})
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

    # Ensure all values are finite
    if not np.all(np.isfinite(path_arr)):
        # Fallback to a simple valid path
        path_arr = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0]
        ], dtype=np.float64)

    return path_arr
