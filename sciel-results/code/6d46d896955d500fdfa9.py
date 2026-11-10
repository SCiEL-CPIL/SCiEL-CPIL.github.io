# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of L-shaped corridor placements to maximize the area
    of the intersection (the sofa that can be moved through the corridor).
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    # Sofa grid points
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = sofa_points.shape[0]
    
    def is_inside_base(u, v):
        """Check if point (u,v) is inside the base L-shaped corridor."""
        # Part 1: 0 <= u <= 1, -3 <= v <= 1
        cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        # Part 2: -3 <= u <= 1, 0 <= v <= 1
        cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return cond1 | cond2
    
    def generate_placements(path):
        """
        Generate all placements including start, end, and interpolated points.
        path: array of shape (n, 3) with (x, y, angle_deg)
        Returns: list of (x, y, angle_rad) tuples
        """
        placements = [(0.0, 0.0, 0.0)]  # Start state
        
        for i in range(len(path) - 1):
            x1, y1, a1 = path[i]
            x2, y2, a2 = path[i + 1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for k in range(1, m + 1):
                t = k / m
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                a = a1 + t * (a2 - a1)
                placements.append((x, y, a))
        
        # Add the last path point if not already added
        if len(path) > 0:
            x, y, a = path[-1]
            placements.append((x, y, a))
        
        # End state
        placements.append((-2.0, 0.0, 90.0))
        
        return placements
    
    def count_inside(points, placements):
        """
        Count how many points are inside ALL placements.
        points: array of shape (n, 2)
        placements: list of (x, y, angle_deg)
        Returns: number of points inside all placements
        """
        # Start with all points as candidates
        mask = np.ones(len(points), dtype=bool)
        
        for (cx, cy, a_deg) in placements:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Rotate points relative to corner
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Check if inside base corridor
            inside = is_inside_base(u, v)
            mask &= inside
            
            # Early exit if no points remain
            if not np.any(mask):
                return 0
        
        return np.sum(mask)
    
    def objective(params):
        """
        Objective function for optimization.
        params: flat array of path parameters
        Returns: negative count (to minimize)
        """
        if time.time() > deadline:
            return 1e10
        
        n_ctrl = len(params) // 3
        path = params.reshape(n_ctrl, 3)
        
        # Ensure angles are monotonic and in reasonable range
        # Clamp angles to [5, 85] to avoid issues
        path[:, 2] = np.clip(path[:, 2], 5, 85)
        
        # Ensure monotonic increase in angle
        for i in range(1, n_ctrl):
            if path[i, 2] <= path[i-1, 2]:
                path[i, 2] = path[i-1, 2] + 0.1
        
        placements = generate_placements(path)
        count = count_inside(sofa_points, placements)
        
        return -count
    
    def objective_with_time(params):
        """Objective with time check."""
        if time.time() > deadline:
            return 1e10
        return objective(params)
    
    # Initial guess: smooth path from near start to near end
    n_ctrl = 6
    angles = np.linspace(10, 80, n_ctrl)
    # Position: arc-like path
    t = np.linspace(0, 1, n_ctrl)
    x0 = -0.1 - 0.6 * t
    y0 = 0.1 + 0.2 * np.sin(np.pi * t)
    
    initial_params = np.column_stack([x0, y0, angles]).ravel()
    
    # Bounds for each parameter
    bounds = []
    for i in range(n_ctrl):
        bounds.append((-1.5, 0.5))  # x
        bounds.append((-0.5, 1.0))  # y
        bounds.append((5, 85))      # angle
    
    # Try differential evolution first (global optimization)
    best_result = None
    best_count = 0
    best_params = initial_params.copy()
    
    # Quick local optimization with Nelder-Mead
    try:
        res = minimize(objective_with_time, initial_params, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4,
                               'adaptive': True})
        if res.fun < 1e9:
            best_count = -res.fun
            best_params = res.x.copy()
            best_result = res
    except Exception as e:
        pass
    
    # Try L-BFGS-B if we have time
    if time.time() < deadline - 100:
        try:
            res2 = minimize(objective_with_time, best_params, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
            if res2.fun < best_count - 0.5:
                best_count = -res2.fun
                best_params = res2.x.copy()
                best_result = res2
        except Exception as e:
            pass
    
    # Try a few random restarts if we have time
    if time.time() < deadline - 50:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            # Random perturbation of best
            noise = np.random.randn(len(initial_params)) * 0.1
            rand_params = best_params + noise
            # Clamp to bounds
            for i in range(len(bounds)):
                rand_params[i] = np.clip(rand_params[i], bounds[i][0], bounds[i][1])
            
            try:
                res_r = minimize(objective_with_time, rand_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3,
                                       'adaptive': True})
                if res_r.fun < best_count - 0.5:
                    best_count = -res_r.fun
                    best_params = res_r.x.copy()
            except Exception as e:
                pass
    
    # Extract best path
    n_ctrl = len(best_params) // 3
    best_path = best_params.reshape(n_ctrl, 3)
    
    # Ensure angles are monotonic
    for i in range(1, n_ctrl):
        if best_path[i, 2] <= best_path[i-1, 2]:
            best_path[i, 2] = best_path[i-1, 2] + 0.1
    
    # Clamp to valid range
    best_path[:, 2] = np.clip(best_path[:, 2], 5, 85)
    
    # Verify the result is valid
    if time.time() < deadline - 10:
        placements = generate_placements(best_path)
        final_count = count_inside(sofa_points, placements)
        
        # If this is worse than the baseline, use baseline
        baseline_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ])
        baseline_placements = generate_placements(baseline_path)
        baseline_count = count_inside(sofa_points, baseline_placements)
        
        if final_count < baseline_count:
            best_path = baseline_path
    
    # Ensure output is finite and valid
    best_path = np.nan_to_num(best_path, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_path


# EVOLVE-BLOCK-END
