# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of (x, y, angle) placements for the L-shaped corridor that
    maximizes the area of the intersection of all corridor placements.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Precompute the sofa grid
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    gx = X.ravel()
    gy = Y.ravel()
    n_pts = len(gx)

    def _check_in_base(u, v):
        """Check if point (u, v) is in the base L-shaped corridor."""
        # Base: {0<=u<=1, -3<=v<=1} UNION {-3<=u<=1, 0<=v<=1}
        in_arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        return in_arm1 | in_arm2

    def _densify_path(path):
        """
        Densify path exactly as evaluator: prepend (0,0,0), append (-2,0,90),
        then interpolate between consecutive points in m = max(1, ceil(|da|)) steps.
        Returns array of shape (N, 3).
        """
        full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
        result = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(math.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                result.append(interp)
        return np.array(result)

    def _evaluate_path(path):
        """
        Evaluate the area of the intersection of all corridor placements.
        Returns the raw value (area) as a float.
        """
        dense = _densify_path(path)
        n_placements = len(dense)

        # For each placement, compute which grid points are inside
        # A grid point is in the sofa if it's inside ALL placements
        mask = np.ones(n_pts, dtype=bool)

        for (cx, cy, a_deg) in dense:
            a_rad = math.radians(a_deg)
            cos_a = math.cos(a_rad)
            sin_a = math.sin(a_rad)

            # Rotate grid points into corridor frame: R(-a)(w - c)
            dx = gx - cx
            dy = gy - cy
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy

            in_base = _check_in_base(u, v)
            mask &= in_base

            # Early exit if no points remain
            if not mask.any():
                return 0.0

        return float(np.sum(mask)) * 0.0001

    # Parameterize the path with a fixed number of interior points
    # We'll use a parameterization that ensures monotonic angle from ~0 to ~90
    # and reasonable positions.

    def _params_to_path(params):
        """
        Convert optimization parameters to a valid path.
        params: array of shape (2*n_interior,) where n_interior is number of interior points.
        For each interior point i, we have (x_i, y_i). Angles are fixed to be evenly spaced.
        """
        n_interior = len(params) // 2
        # Angles evenly spaced between 5 and 85 degrees
        angles = np.linspace(5.0, 85.0, n_interior)
        path = np.zeros((n_interior, 3))
        for i in range(n_interior):
            path[i, 0] = params[2 * i]
            path[i, 1] = params[2 * i + 1]
            path[i, 2] = angles[i]
        return path

    # Start with a reasonable initial path
    n_interior = 8
    x0 = np.array([-0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.3, 0.3,
                   -0.4, 0.3, -0.5, 0.2, -0.6, 0.2, -0.7, 0.3])

    def _objective(params):
        """Negative area for minimization."""
        if time.time() > deadline:
            return 0.0
        path = _params_to_path(params)
        return -_evaluate_path(path)

    # Define bounds for optimization
    # x should be in roughly [-2, 0], y in [0, 1]
    bounds = [(-2.0, 0.5)] * (2 * n_interior)

    # First try a local optimization from the seed
    best_params = x0.copy()
    best_score = -_objective(x0)

    # Local optimization using Nelder-Mead
    res = minimize(_objective, x0, method='Nelder-Mead',
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                            'adaptive': True})
    if res.fun < -best_score:
        best_score = -res.fun
        best_params = res.x

    # Try a few random restarts with local optimization
    np.random.seed(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 60:
            break
        # Random perturbation around the best solution
        noise = np.random.normal(0, 0.15, size=2 * n_interior)
        x_try = best_params + noise
        x_try = np.clip(x_try, bounds[0][0], bounds[0][1])

        res = minimize(_objective, x_try, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6,
                                'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    # Try with different numbers of interior points
    for n_int in [6, 10, 12]:
        if time.time() > deadline - 60:
            break
        # Initialize with interpolation of the best path
        if n_int != n_interior:
            old_path = _params_to_path(best_params)
            # Interpolate to get new number of points
            old_angles = old_path[:, 2]
            new_angles = np.linspace(5.0, 85.0, n_int)
            x_interp = np.interp(new_angles, old_angles, old_path[:, 0])
            y_interp = np.interp(new_angles, old_angles, old_path[:, 1])
            x_try = np.concatenate([x_interp, y_interp])
        else:
            x_try = best_params.copy()

        bounds_n = [(-2.0, 0.5)] * (2 * n_int)

        def _objective_n(params, n_int=n_int):
            if time.time() > deadline:
                return 0.0
            angles = np.linspace(5.0, 85.0, n_int)
            path = np.zeros((n_int, 3))
            path[:, 0] = params[0::2]
            path[:, 1] = params[1::2]
            path[:, 2] = angles
            return -_evaluate_path(path)

        res = minimize(_objective_n, x_try, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-6,
                                'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            # Store the best params for this n_int
            best_params_n = res.x
            best_n_int = n_int

    # Final evaluation
    if time.time() < deadline:
        final_path = _params_to_path(best_params)
    else:
        final_path = _params_to_path(best_params)

    return final_path


# EVOLVE-BLOCK-END
