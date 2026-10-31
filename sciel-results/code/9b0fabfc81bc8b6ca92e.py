"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    """Compute all triangle areas for N points. Returns array of shape (C(N,3),)."""
    n = len(points)
    # Get all combinations of 3 points
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p0 = points[idx[:, 0]]
    p1 = points[idx[:, 1]]
    p2 = points[idx[:, 2]]
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    points = np.asarray(points)
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _objective(x):
    """Objective to minimize: negative score."""
    points = x.reshape(-1, 2)
    return -_score(points)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

def _grid_points(n):
    """Create a grid-like arrangement of n points."""
    # For 13 points, use a 4x4 grid minus one corner, or similar
    # Let's try a pattern: 4x4 grid = 16, remove 3 points
    # Or use a more symmetric arrangement
    # Try: points on concentric rings
    # Ring 0: 1 point at center
    # Ring 1: 6 points (hexagon)
    # Ring 2: 6 points (larger hexagon)
    # Total: 13
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Inner hexagon
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Outer hexagon
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(pts)

def _circle_points(n, radius=1.0):
    """Points evenly spaced on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial configurations
    initial_configs = [
        _regular_polygon(N, radius=1.0),
        _grid_points(N),
        _circle_points(N, radius=1.0),
        # Random points in unit square
        np.random.RandomState(42).rand(N, 2),
        np.random.RandomState(123).rand(N, 2),
    ]
    
    # Evaluate initial configs
    for pts in initial_configs:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Optimization from multiple starting points
    n_opt = 0
    for pts in initial_configs:
        if time.time() > deadline:
            break
        x0 = pts.flatten()
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if time.time() > deadline:
                break
            x_opt = res.x.reshape(-1, 2)
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_points = x_opt.copy()
            n_opt += 1
        except Exception:
            pass
    
    # If we have time, do more random restarts
    rng = np.random.RandomState(0)
    while time.time() < deadline - 5.0:
        # Random points in a circle of radius 1
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        x0 = pts.flatten()
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if time.time() > deadline:
                break
            x_opt = res.x.reshape(-1, 2)
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_points = x_opt.copy()
        except Exception:
            pass
    
    if best_points is None:
        return initial_configs[0]
    
    return best_points
# EVOLVE-BLOCK-END
