"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute minimum triangle area over all C(13,3) = 286 triples."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _make_regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
    """Create n points on a regular polygon."""
    angles = np.linspace(angle_offset, angle_offset + 2*np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _initial_config_7_6():
    """7 outer points on heptagon, 6 inner points on smaller hexagon."""
    outer = _make_regular_polygon(7, radius=1.0, angle_offset=np.pi/2)
    inner = _make_regular_polygon(6, radius=0.45, angle_offset=np.pi/2 + np.pi/6)
    return np.vstack([outer, inner])

def _initial_config_6_7():
    """6 outer points on hexagon, 7 inner points."""
    outer = _make_regular_polygon(6, radius=1.0, angle_offset=np.pi/6)
    # 7 inner: 6 on small hexagon + 1 at center
    inner_hex = _make_regular_polygon(6, radius=0.4, angle_offset=np.pi/6)
    center_pt = np.array([[0.0, 0.0]])
    inner = np.vstack([inner_hex, center_pt])
    return np.vstack([outer, inner])

def _initial_config_5_8():
    """5 outer on pentagon, 8 inner."""
    outer = _make_regular_polygon(5, radius=1.0, angle_offset=np.pi/2)
    # 8 inner: 4 on small square + 4 on smaller square rotated
    inner_sq1 = _make_regular_polygon(4, radius=0.5, angle_offset=np.pi/4)
    inner_sq2 = _make_regular_polygon(4, radius=0.25, angle_offset=0.0)
    inner = np.vstack([inner_sq1, inner_sq2])
    return np.vstack([outer, inner])

def _local_optimize(points, time_limit=10.0):
    """Local optimization using Nelder-Mead to maximize score."""
    start_time = time.time()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    def negative_score(x):
        if time.time() - start_time > time_limit * 0.8:
            return 1e10
        pts = x.reshape(-1, 2)
        s = _score(pts)
        return -s
    
    x0 = points.flatten()
    result = minimize(negative_score, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.fun < -best_score:
        best_points = result.x.reshape(-1, 2)
        best_score = -result.fun
    
    return best_points, best_score

def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try multiple initial configurations
    initializers = [
        _initial_config_7_6,
        _initial_config_6_7,
        _initial_config_5_8,
    ]
    
    best_points = None
    best_score = 0.0
    
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        points = init_func()
        score = _score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
        
        # Local optimization with time check
        if time.time() < deadline - 5.0:
            time_remaining = deadline - time.time() - 2.0
            if time_remaining > 2.0:
                opt_points, opt_score = _local_optimize(points, time_limit=min(time_remaining, 15.0))
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
    
    # Final check
    if best_points is None:
        best_points = _initial_config_7_6()
    
    return best_points
# EVOLVE-BLOCK-END
