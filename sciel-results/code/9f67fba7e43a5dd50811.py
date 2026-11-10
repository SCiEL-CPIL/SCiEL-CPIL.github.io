# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_grid():
    """Precompute grid points for sofa membership testing."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()


def _densify_path(path):
    """
    Densify a path the same way the evaluator does.
    Prepends (0,0,0) and appends (-2,0,90).
    Between consecutive rows, adds linear interpolations in m = max(1, ceil(|a2-a1|)) steps.
    """
    # Prepend start, append end
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    
    # Densify
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)


def _check_in_base(u, v):
    """
    Check if point (u, v) is in the base L-shaped corridor.
    Base: {0 <= u <= 1, -3 <= v <= 1} ∪ {-3 <= u <= 1, 0 <= v <= 1}
    Returns boolean array.
    """
    # Arm 1: 0 <= u <= 1, -3 <= v <= 1
    arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Arm 2: -3 <= u <= 1, 0 <= v <= 1
    arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return arm1 | arm2


def _evaluate_path(path, grid_x, grid_y):
    """
    Evaluate the sofa area for a given path.
    Returns the raw value (area).
    """
    densified = _densify_path(path)
    n_points = len(grid_x)
    
    # Start with all points as potential sofa members
    inside_all = np.ones(n_points, dtype=bool)
    
    for (cx, cy, a_deg) in densified:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) * (w - c)
        # w - c = (grid_x - cx, grid_y - cy)
        dx = grid_x - cx
        dy = grid_y - cy
        
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        # So R(-a)(dx, dy) = (dx*cos(a) + dy*sin(a), -dx*sin(a) + dy*cos(a))
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        in_base = _check_in_base(u, v)
        inside_all &= in_base
        
        # Early exit if no points remain
        if not np.any(inside_all):
            return 0.0
    
    return float(np.sum(inside_all)) * 0.0001


def _nelder_mead_minimize(func, x0, bounds, max_iter=200, tol=1e-6, time_limit=None, start_time=None):
    """
    Simple Nelder-Mead optimizer with bounds and time guard.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    f0 = func(x0)
    simplex.append(x0.copy() + 0.1 * np.ones(n))
    
    # Adjust for bounds
    for i in range(n):
        simplex[1][i] = min(max(simplex[1][i], bounds[i][0]), bounds[i][1])
    
    func_vals = [f0, func(simplex[1])]
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Sort simplex
        order = np.argsort(func_vals)
        simplex = [simplex[i].copy() for i in order]
        func_vals = [func_vals[i] for i in order]
        
        # Check convergence
        if abs(func_vals[0] - func_vals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        # Clip to bounds
        for i in range(n):
            xr[i] = min(max(xr[i], bounds[i][0]), bounds[i][1])
        fr = func(xr)
        
        if fr < func_vals[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            for i in range(n):
                xe[i] = min(max(xe[i], bounds[i][0]), bounds[i][1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                func_vals[-1] = fe
            else:
                simplex[-1] = xr
                func_vals[-1] = fr
        elif fr < func_vals[-2]:
            simplex[-1] = xr
            func_vals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            for i in range(n):
                xc[i] = min(max(xc[i], bounds[i][0]), bounds[i][1])
            fc = func(xc)
            if fc < func_vals[-1]:
                simplex[-1] = xc
                func_vals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    func_vals[i] = func(simplex[i])
    
    # Return best point
    best_idx = np.argmin(func_vals)
    return simplex[best_idx], func_vals[best_idx]


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the moving sofa path to maximize the sofa area.
    """
    start_time = time.time()
    time_limit = 580  # Leave margin for safety
    
    # Build grid
    grid_x, grid_y = _build_grid()
    
    # Initial path (the baseline)
    best_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_score = _evaluate_path(best_path, grid_x, grid_y)
    
    # Flatten path for optimization: 8 * 3 = 24 parameters
    # Bounds: x in [-2, 1], y in [-1, 1], a in [0, 90]
    bounds = []
    for _ in range(8):
        bounds.append((-2.0, 1.0))  # x
        bounds.append((-1.0, 1.0))  # y
        bounds.append((5.0, 85.0))  # a (avoid 0 and 90 since endpoints are fixed)
    
    def objective(params):
        path = params.reshape(8, 3)
        # Ensure angles are monotonically increasing
        for i in range(1, 8):
            if path[i, 2] < path[i-1, 2]:
                path[i, 2] = path[i-1, 2]
        return -_evaluate_path(path, grid_x, grid_y)
    
    # Run Nelder-Mead
    x0 = best_path.ravel()
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 30:
        result, score = _nelder_mead_minimize(
            objective, x0, bounds,
            max_iter=100, tol=1e-5,
            time_limit=remaining_time - 10,
            start_time=start_time
        )
        candidate_path = result.reshape(8, 3)
        candidate_score = -score
        if candidate_score > best_score:
            best_path = candidate_path
            best_score = candidate_score
    
    # Try a few random perturbations if time allows
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() - start_time > time_limit - 30:
            break
        perturbation = rng.normal(0, 0.05, size=(8, 3))
        perturbation[:, 2] = np.abs(perturbation[:, 2]) * 2  # Larger angle perturbations
        candidate = best_path + perturbation
        # Clip to bounds
        candidate[:, 0] = np.clip(candidate[:, 0], -2.0, 1.0)
        candidate[:, 1] = np.clip(candidate[:, 1], -1.0, 1.0)
        candidate[:, 2] = np.clip(candidate[:, 2], 5.0, 85.0)
        # Ensure monotonic angles
        for i in range(1, 8):
            if candidate[i, 2] < candidate[i-1, 2]:
                candidate[i, 2] = candidate[i-1, 2]
        
        cand_score = _evaluate_path(candidate, grid_x, grid_y)
        if cand_score > best_score:
            best_path = candidate
            best_score = cand_score
    
    return best_path


# EVOLVE-BLOCK-END
