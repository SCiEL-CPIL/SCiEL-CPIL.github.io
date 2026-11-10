# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Time budget: return before 630s wall-clock, keep safety margin ~20s
TIME_LIMIT = 610.0
START_TIME = time.time()


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _base_mask_fast(x, y, a_deg):
    """
    Fast check: does the base corridor (at (x,y,a)) contain all sofa points?
    We test each sofa point against the rotated base corridor.
    
    Base corridor = union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
    A point w is inside placement (x,y,a) if R(-a)(w - (x,y)) is in base.
    
    For speed, we transform all sofa points at once.
    """
    a_rad = np.deg2rad(a_deg)
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    # R(-a) = [[cos, sin], [-sin, cos]]
    # u = ca*(wx - x) + sa*(wy - y)
    # v = -sa*(wx - x) + ca*(wy - y)
    # We'll precompute sofa points as (wx, wy) arrays passed in.
    pass  # placeholder; actual computation done in score_path


def _sofa_points():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return gx.ravel(), gy.ravel()


def _check_placement(x, y, a_deg, gx, gy):
    """
    Check if all sofa points are inside the corridor placement (x, y, a_deg).
    Returns True if all points are inside.
    """
    a_rad = np.deg2rad(a_deg)
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    # R(-a) applied to (w - (x,y)):
    # u = ca*(gx - x) + sa*(gy - y)
    # v = -sa*(gx - x) + ca*(gy - y)
    u = ca * (gx - x) + sa * (gy - y)
    v = -sa * (gx - x) + ca * (gy - y)
    
    # Base corridor: union of rect1 = {0<=u<=1, -3<=v<=1} and rect2 = {-3<=u<=1, 0<=v<=1}
    in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    inside = in_rect1 | in_rect2
    return np.all(inside)


def _score_path(path, gx, gy):
    """
    Score a path: the sofa is the set of points inside EVERY placement.
    We check each placement; if any point fails any placement, it's excluded.
    Score = (number of points in intersection) * 0.0001
    """
    n_points = len(gx)
    mask = np.ones(n_points, dtype=bool)
    
    for (x, y, a) in path:
        a_rad = np.deg2rad(a)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        u = ca * (gx - x) + sa * (gy - y)
        v = -sa * (gx - x) + ca * (gy - y)
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        inside = in_rect1 | in_rect2
        mask &= inside
        # Early exit if no points remain
        if not np.any(mask):
            return 0.0
    
    return float(np.sum(mask)) * 0.0001


def _densify_path(path):
    """
    Insert linear interpolations between consecutive rows.
    m = max(1, ceil(|a2 - a1|)) equal steps.
    This produces m+1 points including endpoints? Actually "adds linear interpolations in m equal steps"
    means between row i and row i+1, we add m-1 intermediate points (total m+1 including endpoints).
    Wait: "adds linear interpolations of (x,y,a) in m = max(1, ceil(|a2-a1|)) equal steps"
    This means we split the segment into m equal sub-segments, giving m+1 points total (including both ends).
    So we add m-1 new points between each pair.
    """
    result = [path[0].copy()]
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        m = max(1, int(np.ceil(abs(p2[2] - p1[2]))))
        # m equal steps -> m+1 points, we already have p1, add m-1 intermediates and p2
        for k in range(1, m):
            t = k / m
            pt = p1 + t * (p2 - p1)
            result.append(pt)
        result.append(p2)
    return np.array(result)


def _evaluate_full_path(path, gx, gy):
    """
    Prepend (0,0,0), append (-2,0,90), densify, then score.
    """
    full = np.vstack([
        np.array([0.0, 0.0, 0.0]),
        path,
        np.array([-2.0, 0.0, 90.0])
    ])
    full = _densify_path(full)
    return _score_path(full, gx, gy)


def _objective(params, gx, gy):
    """
    Objective for minimization: negative score.
    params: 1D array of length 3*n, representing n path points (x,y,a) each.
    We need to ensure angles are in a sensible range and the path is monotonic in angle.
    """
    n = len(params) // 3
    path = params.reshape(n, 3)
    # Clamp angles to [1, 89] to stay between 0 and 90
    path[:, 2] = np.clip(path[:, 2], 1.0, 89.0)
    # Ensure monotonic increasing angle
    for i in range(1, n):
        if path[i, 2] <= path[i-1, 2]:
            path[i, 2] = path[i-1, 2] + 0.1
    # Clamp x,y to reasonable range
    path[:, 0] = np.clip(path[:, 0], -3.0, 3.0)
    path[:, 1] = np.clip(path[:, 1], -3.0, 3.0)
    score = _evaluate_full_path(path, gx, gy)
    return -score


def _bounds(n):
    """Bounds for L-BFGS-B: x in [-3,3], y in [-3,3], a in [1,89]"""
    bounds = []
    for _ in range(n):
        bounds.append((-3.0, 3.0))
        bounds.append((-3.0, 3.0))
        bounds.append((1.0, 89.0))
    return bounds


def _perturb(path, scale=0.5):
    """Randomly perturb a path."""
    new_path = path.copy()
    new_path[:, 0] += np.random.normal(0, scale, size=new_path.shape[0])
    new_path[:, 1] += np.random.normal(0, scale, size=new_path.shape[0])
    new_path[:, 2] += np.random.normal(0, scale, size=new_path.shape[0])
    # Enforce monotonic angles
    new_path[:, 2] = np.clip(new_path[:, 2], 1.0, 89.0)
    for i in range(1, len(new_path)):
        if new_path[i, 2] <= new_path[i-1, 2]:
            new_path[i, 2] = new_path[i-1, 2] + 0.1
    return new_path


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path that maximizes the area of the sofa that can be moved through the L-corridor.
    Uses L-BFGS-B optimization with multiple restarts and coordinate ascent polish.
    """
    gx, gy = _sofa_points()
    
    # Baseline path (from seed)
    baseline = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_path = baseline.copy()
    best_score = _evaluate_full_path(best_path, gx, gy)
    
    n_points = 8  # Number of interior path points
    
    # Try multiple restarts with L-BFGS-B
    n_restarts = 20
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if _remaining_time() < 30:
            break
        
        if restart == 0:
            # Start from baseline
            x0 = best_path.ravel()
        else:
            # Start from perturbed best or random
            if restart % 2 == 0:
                x0 = _perturb(best_path, scale=1.0).ravel()
            else:
                # Random initialization with monotonic angles
                angles = np.sort(np.random.uniform(5, 85, size=n_points))
                xs = np.random.uniform(-2, 0, size=n_points)
                ys = np.random.uniform(0, 1, size=n_points)
                x0 = np.column_stack([xs, ys, angles]).ravel()
        
        try:
            res = minimize(
                _objective, x0, args=(gx, gy),
                method='L-BFGS-B',
                bounds=_bounds(n_points),
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-6, 'gtol': 1e-4}
            )
        except Exception:
            continue
        
        if _remaining_time() < 25:
            break
        
        # Evaluate the result
        res_path = res.x.reshape(n_points, 3)
        res_path[:, 2] = np.clip(res_path[:, 2], 1.0, 89.0)
        for i in range(1, n_points):
            if res_path[i, 2] <= res_path[i-1, 2]:
                res_path[i, 2] = res_path[i-1, 2] + 0.1
        res_path[:, 0] = np.clip(res_path[:, 0], -3.0, 3.0)
        res_path[:, 1] = np.clip(res_path[:, 1], -3.0, 3.0)
        
        res_score = _evaluate_full_path(res_path, gx, gy)
        
        if res_score > best_score:
            best_score = res_score
            best_path = res_path.copy()
            print(f"Restart {restart}: improved to {best_score:.6f}")
    
    # Coordinate ascent polish
    if _remaining_time() > 20:
        for _ in range(50):
            if _remaining_time() < 15:
                break
            improved = False
            for i in range(n_points):
                for dim in range(3):
                    old_val = best_path[i, dim]
                    for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                        test_path = best_path.copy()
                        test_path[i, dim] = old_val + delta
                        # Enforce constraints
                        test_path[:, 2] = np.clip(test_path[:, 2], 1.0, 89.0)
                        for j in range(1, n_points):
                            if test_path[j, 2] <= test_path[j-1, 2]:
                                test_path[j, 2] = test_path[j-1, 2] + 0.1
                        test_path[:, 0] = np.clip(test_path[:, 0], -3.0, 3.0)
                        test_path[:, 1] = np.clip(test_path[:, 1], -3.0, 3.0)
                        
                        test_score = _evaluate_full_path(test_path, gx, gy)
                        if test_score > best_score:
                            best_score = test_score
                            best_path = test_path.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Final safety check
    if _remaining_time() < 5:
        pass  # Just return
    
    return best_path


# EVOLVE-BLOCK-END
