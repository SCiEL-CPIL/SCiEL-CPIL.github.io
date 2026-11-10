# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _build_grid():
    """Precompute grid points for sofa membership testing."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()


def _check_in_base(u, v):
    """
    Check if point (u, v) is inside the base corridor.
    Base corridor is union of:
      {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    """
    # Rectangle 1: 0 <= u <= 1, -3 <= v <= 1
    in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Rectangle 2: -3 <= u <= 1, 0 <= v <= 1
    in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return in_rect1 | in_rect2


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


def _evaluate_path(path, grid_x, grid_y):
    """
    Evaluate the area of the sofa for a given path.
    The sofa is the set of grid points that are inside every placement.
    """
    # Densify the path
    densified = _densify_path(path)
    
    # Get grid points
    x = grid_x
    y = grid_y
    
    # Initialize mask with True for all points
    mask = np.ones(len(x), dtype=bool)
    
    # For each placement, check which points are inside
    for px, py, angle_deg in densified:
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)
        
        # Rotation matrix R(-angle): rotate points by -angle
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        
        # Translate and rotate: w' = R(-a)(w - (px, py))
        dx = x - px
        dy = y - py
        u = c * dx + s * dy
        v = -s * dx + c * dy
        
        # Check if in base corridor
        in_base = _check_in_base(u, v)
        
        # Update mask: point must be in ALL placements
        mask &= in_base
        
        # Early exit if no points remain
        if not mask.any():
            return 0.0
    
    # Count remaining points
    count = mask.sum()
    return float(count) * 0.0001


def _objective(params, grid_x, grid_y):
    """
    Objective function for optimization: returns negative area (to minimize).
    params: flat array of [x1, y1, a1, x2, y2, a2, ..., xn, yn, an]
    """
    n = len(params) // 3
    path = params.reshape(n, 3)
    area = _evaluate_path(path, grid_x, grid_y)
    return -area


def _optimize(n_points, grid_x, grid_y, deadline, initial_path=None):
    """
    Use differential evolution to optimize the path.
    """
    # Bounds for each parameter
    # x: roughly -2 to 1, y: -1 to 1, angle: 0 to 90
    bounds = []
    for _ in range(n_points):
        bounds.append((-2.0, 1.0))   # x
        bounds.append((-1.0, 1.0))   # y
        bounds.append((0.0, 90.0))   # angle
    
    # Initial population
    popsize = 10
    if initial_path is not None:
        # Include initial path in the initial population
        init_pop = [initial_path.flatten()]
        # Add random initial points
        for _ in range(popsize - 1):
            init_pop.append(np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            ))
        init = np.array(init_pop)
    else:
        init = 'random'
    
    # Optimize
    result = differential_evolution(
        _objective,
        bounds,
        args=(grid_x, grid_y),
        init=init,
        maxiter=50,
        tol=1e-6,
        popsize=popsize,
        seed=42,
        polish=False,
        callback=lambda xk, convergence: (
            time.time() > deadline - 60
        )
    )
    
    best_params = result.x
    best_path = best_params.reshape(n_points, 3)
    best_area = _evaluate_path(best_path, grid_x, grid_y)
    
    return best_path, best_area


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline with safety margin
    
    # Precompute grid
    grid_x, grid_y = _build_grid()
    
    # Start with the baseline path
    baseline_path = np.array(
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
    
    # Evaluate baseline
    baseline_area = _evaluate_path(baseline_path, grid_x, grid_y)
    best_path = baseline_path
    best_area = baseline_area
    
    # Try different numbers of control points
    for n_points in [5, 8, 10, 12]:
        if time.time() > deadline - 120:
            break
        
        # Create initial path with n_points
        if n_points == 8:
            initial = baseline_path
        else:
            # Interpolate or create a reasonable initial path
            t = np.linspace(0, 1, n_points)
            # Use a rough arc as initial guess
            x_init = np.linspace(-0.1, -0.7, n_points)
            y_init = np.linspace(0.1, 0.3, n_points)
            a_init = np.linspace(15, 86, n_points)
            initial = np.column_stack([x_init, y_init, a_init])
        
        try:
            path, area = _optimize(n_points, grid_x, grid_y, deadline, initial_path=initial)
            if area > best_area:
                best_path = path
                best_area = area
        except Exception as e:
            print(f"Optimization failed for n_points={n_points}: {e}")
            continue
    
    # Final polish: local optimization with tighter tolerance
    if time.time() < deadline - 60 and best_area > 0:
        from scipy.optimize import minimize
        
        # Flatten best path
        x0 = best_path.flatten()
        
        def obj_local(params):
            path = params.reshape(len(x0) // 3, 3)
            return -_evaluate_path(path, grid_x, grid_y)
        
        try:
            result = minimize(
                obj_local,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            polished_path = result.x.reshape(len(x0) // 3, 3)
            polished_area = _evaluate_path(polished_path, grid_x, grid_y)
            if polished_area > best_area:
                best_path = polished_path
                best_area = polished_area
        except Exception as e:
            print(f"Polish failed: {e}")
    
    # Ensure the path is valid
    if not np.all(np.isfinite(best_path)):
        best_path = baseline_path
    
    return best_path


# EVOLVE-BLOCK-END
