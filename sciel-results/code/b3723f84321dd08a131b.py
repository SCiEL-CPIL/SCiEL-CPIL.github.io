"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    # Using cross products / area method
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def _project_inside(points):
    """Project points that are slightly outside back inside the triangle."""
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    for i in range(len(points)):
        p = points[i]
        if not _is_inside_triangle(p, eps=1e-6):
            # Find closest point on triangle boundary
            # Simple approach: move towards centroid
            centroid = (v0 + v1 + v2) / 3
            # Binary search for the closest valid point
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                candidate = p + mid * (centroid - p)
                if _is_inside_triangle(candidate, eps=1e-9):
                    lo = mid
                else:
                    hi = mid
            points[i] = p + lo * (centroid - p)
    return points

def _objective(x):
    """Objective function: negative of minimum triangle area (to minimize)."""
    points = x.reshape(N, 2)
    # Penalize points outside the triangle
    penalty = 0.0
    for i in range(N):
        if not _is_inside_triangle(points[i], eps=-1e-3):
            # Compute distance outside
            v0 = np.array([0.0, 0.0])
            v1 = np.array([1.0, 0.0])
            v2 = np.array([0.5, np.sqrt(3) / 2])
            # Approximate penalty
            penalty += 100.0
    area = _min_triangle_area(points)
    return -area + penalty

def _make_initial_config_1():
    """Symmetric configuration: 3 vertices, 3 edge midpoints, 3 quarter points, 2 interior."""
    h = np.sqrt(3) / 2
    pts = [
        [0.0, 0.0],           # vertex 0
        [1.0, 0.0],           # vertex 1
        [0.5, h],             # vertex 2
        [0.5, 0.0],           # midpoint of edge 0-1
        [0.75, h/2],          # midpoint of edge 1-2
        [0.25, h/2],          # midpoint of edge 2-0
        [0.25, 0.0],          # quarter point on edge 0-1
        [0.75, 0.0],          # three-quarter on edge 0-1
        [0.5, h*0.25],        # interior point 1
        [0.3, h*0.3],         # interior point 2
        [0.7, h*0.3],         # interior point 3
    ]
    return np.array(pts)

def _make_initial_config_2():
    """Hexagonal-like arrangement."""
    h = np.sqrt(3) / 2
    # Use barycentric coordinates for a more symmetric layout
    # 3 corners, 3 near-midpoints, 5 interior in a pattern
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.5, h/2],
        [0.165, h/4],
        [0.835, h/4],
        [0.33, h/3],
        [0.67, h/3],
        [0.5, h*2/3],
    ]
    return np.array(pts)

def _make_initial_config_3():
    """Grid-like arrangement in barycentric coords."""
    h = np.sqrt(3) / 2
    # Points at various barycentric positions
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, h*0.25],
        [0.125, h*0.25],
        [0.875, h*0.25],
        [0.375, h*0.5],
        [0.625, h*0.5],
        [0.5, h*0.75],
    ]
    return np.array(pts)

def _make_random_config(seed):
    """Generate a random configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = [u + 0.5 * v, (np.sqrt(3) / 2) * v]
                pts.append(p)
                break
    return np.array(pts)

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    initial_configs = [
        _make_initial_config_1(),
        _make_initial_config_2(),
        _make_initial_config_3(),
    ]
    
    # Add a few random configs
    for s in [42, 123, 456]:
        initial_configs.append(_make_random_config(s))
    
    best_points = None
    best_area = -float('inf')
    
    for config in initial_configs:
        if time.time() > deadline - 20.0:
            break
        
        # Project inside
        config = _project_inside(config.copy())
        
        # Evaluate initial
        area = _min_triangle_area(config)
        if area > best_area:
            best_area = area
            best_points = config.copy()
        
        # Try to optimize
        try:
            result = minimize(
                _objective,
                config.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if time.time() > deadline - 10.0:
                break
            
            opt_points = result.x.reshape(N, 2)
            opt_points = _project_inside(opt_points)
            
            # Verify all points are inside
            valid = True
            for i in range(N):
                if not _is_inside_triangle(opt_points[i], eps=1e-6):
                    valid = False
                    break
            
            if valid:
                opt_area = _min_triangle_area(opt_points)
                if opt_area > best_area:
                    best_area = opt_area
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # If we have time, do a second round of refinement on the best
    if best_points is not None and time.time() < deadline - 30.0:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            opt_points = result.x.reshape(N, 2)
            opt_points = _project_inside(opt_points)
            
            valid = True
            for i in range(N):
                if not _is_inside_triangle(opt_points[i], eps=1e-6):
                    valid = False
                    break
            
            if valid:
                opt_area = _min_triangle_area(opt_points)
                if opt_area > best_area:
                    best_area = opt_area
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to the original random approach
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
    
    return best_points
# EVOLVE-BLOCK-END
