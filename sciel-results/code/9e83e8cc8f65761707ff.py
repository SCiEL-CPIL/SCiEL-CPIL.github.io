# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def _elapsed(start_time):
    return time.time() - start_time

def _check_deadline(start_time, remaining=30.0):
    """Return True if we should stop searching (deadline approaching)."""
    return _elapsed(start_time) > 630.0 - remaining

def _point_in_base(u, v):
    """Vectorized check if (u, v) is in the base L-shaped corridor.
    Base is union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}."""
    cond1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
    cond2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
    return cond1 | cond2

def _generate_placements(path):
    """
    Given a path (n, 3) array of (x, y, a) in degrees, generate all placements
    including endpoints (0,0,0) and (-2,0,90), with linear interpolation.
    Returns array of shape (m, 3) with all placements.
    """
    # Prepend start, append end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    all_points = [start]
    for i in range(len(path)):
        all_points.append(path[i])
    all_points.append(end)
    all_points = np.array(all_points, dtype=np.float64)
    
    # Now densify: between consecutive points, interpolate
    placements = [all_points[0]]
    for i in range(len(all_points) - 1):
        p1 = all_points[i]
        p2 = all_points[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(math.ceil(da)))
        if m > 1:
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                placements.append(interp)
        else:
            placements.append(p2)
    
    return np.array(placements, dtype=np.float64)

def _calculate_area(path):
    """
    Calculate the area of the sofa (intersection of all corridor placements).
    Returns the raw value: number of sofa grid points * 0.0001.
    """
    placements = _generate_placements(path)
    
    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    n_points = len(points)
    
    # For each placement, check which points are inside
    # A point is in the sofa if it's inside ALL placements
    inside_all = np.ones(n_points, dtype=bool)
    
    for px, py, pa in placements:
        a_rad = math.radians(pa)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        
        # Rotate points by -a around (px, py)
        dx = points[:, 0] - px
        dy = points[:, 1] - py
        
        # R(-a) * (w - (x,y))
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        in_base = _point_in_base(u, v)
        inside_all = inside_all & in_base
        
        # Early exit if no points remain
        if not np.any(inside_all):
            break
    
    count = np.sum(inside_all)
    return count * 0.0001

def _objective(params, path_template):
    """
    Objective function for optimization. params is a flat array of path coordinates.
    Returns negative area (for minimization).
    """
    n = len(path_template)
    path = params.reshape(n, 3)
    
    # Ensure angles are in reasonable range
    path[:, 2] = np.clip(path[:, 2], 1.0, 89.0)
    
    try:
        area = _calculate_area(path)
        return -area
    except Exception:
        return 1.0  # Large penalty

def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    """
    start_time = time.time()
    
    # Initial path: a curved path that stays closer to the overlap region
    # Angles go from ~15 to ~86 degrees
    initial_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_path = initial_path.copy()
    best_area = _calculate_area(best_path)
    
    # Random restarts with growing perturbation
    n_restarts = 15
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if _check_deadline(start_time, remaining=60.0):
            break
        
        # Perturb the initial path
        scale = 0.05 + 0.02 * restart
        perturbed = initial_path.copy()
        perturbed[:, 0] += rng.normal(0, scale, len(perturbed))
        perturbed[:, 1] += rng.normal(0, scale, len(perturbed))
        perturbed[:, 2] += rng.normal(0, scale * 2, len(perturbed))
        perturbed[:, 2] = np.clip(perturbed[:, 2], 5.0, 85.0)
        
        # Use scipy minimize
        try:
            result = minimize(
                _objective,
                perturbed.ravel(),
                args=(initial_path,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if result.success or result.fun < -best_area:
                candidate_path = result.x.reshape(len(initial_path), 3)
                candidate_path[:, 2] = np.clip(candidate_path[:, 2], 1.0, 89.0)
                candidate_area = _calculate_area(candidate_path)
                
                if candidate_area > best_area:
                    best_area = candidate_area
                    best_path = candidate_path.copy()
        except Exception:
            pass
    
    # Final polish phase with tighter tolerance
    if not _check_deadline(start_time, remaining=30.0):
        try:
            result = minimize(
                _objective,
                best_path.ravel(),
                args=(initial_path,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < -best_area:
                candidate_path = result.x.reshape(len(initial_path), 3)
                candidate_path[:, 2] = np.clip(candidate_path[:, 2], 1.0, 89.0)
                candidate_area = _calculate_area(candidate_path)
                
                if candidate_area > best_area:
                    best_area = candidate_area
                    best_path = candidate_path.copy()
        except Exception:
            pass
    
    return best_path

# EVOLVE-BLOCK-END
