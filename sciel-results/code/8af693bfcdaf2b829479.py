# EVOLVE-BLOCK-START
import numpy as np
import time


def _check_inside(points, x, y, a_deg):
    """Check which points are inside the corridor placement."""
    a = np.radians(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    in_h = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_v = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return in_h | in_v


def _evaluate_path_fast(path, grid_points):
    """Evaluate the number of grid points inside all placements."""
    inside = np.ones(len(grid_points), dtype=bool)
    for i in range(len(path)):
        x, y, a = path[i]
        inside &= _check_inside(grid_points, x, y, a)
        if not inside.any():
            break
    return np.sum(inside)


def _build_full_path(path):
    """Build the full path with endpoints and interpolation."""
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    full = [start]
    prev = start
    for p in path:
        p = np.array(p)
        da = abs(p[2] - prev[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = prev + t * (p - prev)
            full.append(interp)
        prev = p
    full.append(end)
    return np.array(full)


def moving_sofa_2d() -> np.ndarray:
    """Returns a path of shape (n, 3) that maximizes the sofa area."""
    deadline = time.time() + 580
    start_time = time.time()
    
    # Use a coarser grid for fast evaluation, then verify with fine grid
    xs_coarse = np.arange(-3, 1.5, 0.05)
    ys_coarse = np.arange(-1.5, 1.5, 0.05)
    GXc, GYc = np.meshgrid(xs_coarse, ys_coarse, indexing='ij')
    grid_coarse = np.column_stack([GXc.ravel(), GYc.ravel()])
    
    # Baseline path
    path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    full_path = _build_full_path(path)
    best_count_coarse = _evaluate_path_fast(full_path, grid_coarse)
    best_path = path.copy()
    
    np.random.seed(42)
    
    # Phase 1: Coarse optimization
    while time.time() < deadline - 120:
        improved = False
        for i in range(len(path)):
            if time.time() > deadline - 120:
                break
            for attempt in range(30):
                pert = np.random.randn(3) * 0.15
                new_path = path.copy()
                new_path[i] = path[i] + pert
                new_path[i, 2] = np.clip(new_path[i, 2], 1, 89)
                new_path[i, 0] = np.clip(new_path[i, 0], -3, 1)
                new_path[i, 1] = np.clip(new_path[i, 1], -2, 2)
                
                full_p = _build_full_path(new_path)
                count = _evaluate_path_fast(full_p, grid_coarse)
                
                if count > best_count_coarse:
                    best_count_coarse = count
                    best_path = new_path.copy()
                    path = new_path.copy()
                    improved = True
                    break
        
        # Occasionally try adding/removing points
        if not improved and len(path) < 20:
            if np.random.rand() < 0.3:
                idx = np.random.randint(len(path))
                new_point = path[idx] + np.random.randn(3) * 0.2
                new_point[2] = np.clip(new_point[2], 1, 89)
                new_path = np.vstack([path[:idx+1], new_point, path[idx+1:]])
                full_p = _build_full_path(new_path)
                count = _evaluate_path_fast(full_p, grid_coarse)
                if count > best_count_coarse:
                    best_count_coarse = count
                    best_path = new_path.copy()
                    path = new_path.copy()
        elif not improved and len(path) > 3:
            if np.random.rand() < 0.2:
                idx = np.random.randint(len(path))
                new_path = np.delete(path, idx, axis=0)
                full_p = _build_full_path(new_path)
                count = _evaluate_path_fast(full_p, grid_coarse)
                if count > best_count_coarse:
                    best_count_coarse = count
                    best_path = new_path.copy()
                    path = new_path.copy()
    
    # Phase 2: Fine optimization with the actual grid
    xs_fine = np.arange(-3, 1.5, 0.01)
    ys_fine = np.arange(-1.5, 1.5, 0.01)
    GXf, GYf = np.meshgrid(xs_fine, ys_fine, indexing='ij')
    grid_fine = np.column_stack([GXf.ravel(), GYf.ravel()])
    
    full_path = _build_full_path(best_path)
    best_count_fine = _evaluate_path_fast(full_path, grid_fine)
    
    while time.time() < deadline - 20:
        improved = False
        for i in range(len(best_path)):
            if time.time() > deadline - 20:
                break
            for attempt in range(50):
                pert = np.random.randn(3) * 0.05
                new_path = best_path.copy()
                new_path[i] = best_path[i] + pert
                new_path[i, 2] = np.clip(new_path[i, 2], 1, 89)
                new_path[i, 0] = np.clip(new_path[i, 0], -3, 1)
                new_path[i, 1] = np.clip(new_path[i, 1], -2, 2)
                
                full_p = _build_full_path(new_path)
                count = _evaluate_path_fast(full_p, grid_fine)
                
                if count > best_count_fine:
                    best_count_fine = count
                    improved = True
                    best_path = new_path.copy()
                    break
        
        if not improved:
            break
    
    return best_path


# EVOLVE-BLOCK-END
