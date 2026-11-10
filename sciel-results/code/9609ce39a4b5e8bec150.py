# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


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

    def _evaluate_path_vectorized(path):
        """
        Vectorized evaluation: compute for all placements at once using broadcasting.
        Returns the raw value (area) as a float.
        """
        dense = _densify_path(path)
        n_placements = len(dense)
        
        # Extract placement parameters
        cx = dense[:, 0]
        cy = dense[:, 1]
        a_deg = dense[:, 2]
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Compute rotated coordinates for all grid points and all placements
        # Shape: (n_placements, n_pts)
        dx = gx[None, :] - cx[:, None]
        dy = gy[None, :] - cy[:, None]
        u = cos_a[:, None] * dx + sin_a[:, None] * dy
        v = -sin_a[:, None] * dx + cos_a[:, None] * dy
        
        # Check base corridor membership for all points
        in_arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        in_base = in_arm1 | in_arm2  # Shape: (n_placements, n_pts)
        
        # A point is in the sofa if it's in ALL placements
        mask = in_base.all(axis=0)  # Shape: (n_pts,)
        
        return float(np.sum(mask)) * 0.0001

    def _evaluate_path(path):
        """
        Evaluate the area of the intersection of all corridor placements.
        Returns the raw value (area) as a float.
        """
        return _evaluate_path_vectorized(path)

    # Parameterize the path with a fixed number of interior points
    def _params_to_path(params, n_interior=None):
        """
        Convert optimization parameters to a valid path.
        params: array of shape (2*n_interior,) where n_interior is number of interior points.
        For each interior point i, we have (x_i, y_i). Angles are fixed to be evenly spaced.
        """
        if n_interior is None:
            n_interior = len(params) // 2
        # Angles evenly spaced between 5 and 85 degrees
        angles = np.linspace(5.0, 85.0, n_interior)
        path = np.zeros((n_interior, 3))
        path[:, 0] = params[0::2]
        path[:, 1] = params[1::2]
        path[:, 2] = angles
        return path

    # Start with a reasonable initial path
    n_interior = 10
    # Better initial guess based on known good paths
    x0 = np.array([-0.05, 0.05, -0.15, 0.15, -0.25, 0.25, -0.35, 0.35,
                   -0.45, 0.35, -0.55, 0.30, -0.60, 0.25, -0.65, 0.20,
                   -0.70, 0.15, -0.75, 0.10])

    def _objective(params, n_int=None):
        """Negative area for minimization."""
        if time.time() > deadline:
            return 0.0
        if n_int is None:
            n_int = len(params) // 2
        path = _params_to_path(params, n_int)
        return -_evaluate_path(path)

    # Define bounds for optimization
    bounds = [(-2.0, 0.5)] * (2 * n_interior)

    # First try a local optimization from the seed
    best_params = x0.copy()
    best_score = -_objective(x0, n_interior)

    # Local optimization using Nelder-Mead
    res = minimize(_objective, x0, args=(n_interior,), method='Nelder-Mead',
                   options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-7,
                            'adaptive': True})
    if res.fun < -best_score:
        best_score = -res.fun
        best_params = res.x

    # Try more random restarts with local optimization
    np.random.seed(42)
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 90:
            break
        # Random perturbation around the best solution
        sigma = 0.12 if r < 10 else 0.08
        noise = np.random.normal(0, sigma, size=2 * n_interior)
        x_try = best_params + noise
        x_try = np.clip(x_try, bounds[0][0], bounds[0][1])

        res = minimize(_objective, x_try, args=(n_interior,), method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-7,
                                'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    # Try with different numbers of interior points
    for n_int in [8, 12, 14]:
        if time.time() > deadline - 90:
            break
        # Initialize with interpolation of the best path
        if n_int != n_interior:
            old_path = _params_to_path(best_params, n_interior)
            old_angles = old_path[:, 2]
            new_angles = np.linspace(5.0, 85.0, n_int)
            x_interp = np.interp(new_angles, old_angles, old_path[:, 0])
            y_interp = np.interp(new_angles, old_angles, old_path[:, 1])
            x_try = np.empty(2 * n_int)
            x_try[0::2] = x_interp
            x_try[1::2] = y_interp
        else:
            x_try = best_params.copy()

        res = minimize(_objective, x_try, args=(n_int,), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7,
                                'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
            n_interior = n_int

    # Final polish with tighter tolerances
    if time.time() < deadline - 30:
        res = minimize(_objective, best_params, args=(n_interior,), method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8,
                                'adaptive': True})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    final_path = _params_to_path(best_params, n_interior)
    return final_path


# EVOLVE-BLOCK-END
