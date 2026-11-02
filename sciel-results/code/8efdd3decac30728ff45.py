"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import random


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Find the point with lowest y (and lowest x in case of tie)
    start = 0
    for i in range(1, n):
        if points[i][1] < points[start][1] or (points[i][1] == points[start][1] and points[i][0] < points[start][0]):
            start = i
    
    def angle_cmp(p):
        return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
    
    sorted_pts = sorted(points, key=angle_cmp)
    sorted_pts.insert(0, points[start])
    
    hull = []
    # Build lower hull
    for p in sorted_pts:
        while len(hull) >= 2:
            p1 = hull[-2]
            p2 = hull[-1]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(p)
    
    # Build upper hull
    lower_len = len(hull)
    for p in sorted_pts[-2::-1]:
        while len(hull) > lower_len:
            p1 = hull[-2]
            p2 = hull[-1]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(p)
    
    if len(hull) > 1:
        hull.pop()
    
    # Compute area using shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) * 0.5


def _score(points):
    """Compute the minimum triangle area divided by convex hull area."""
    n = len(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _objective(x):
    """Negative of the score for minimization."""
    points = x.reshape(N, 2)
    return -_score(points)


def _initial_points_regular_polygon():
    """Regular polygon initial configuration."""
    angles = 2 * np.pi * np.arange(N) / N
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _initial_points_concentric_rings():
    """Concentric rings initial configuration."""
    # Outer ring: 7 points, inner ring: 6 points, center: 1 point
    points = []
    # Center
    points.append([0.0, 0.0])
    # Inner ring (6 points at radius 0.5)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Outer ring (7 points at radius 1.0)
    for i in range(7):
        angle = 2 * np.pi * i / 7
        points.append([np.cos(angle), np.sin(angle)])
    return np.array(points)


def _initial_points_grid():
    """Grid-like initial configuration."""
    # 4x4 grid with some perturbation
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if len(points) < N:
                points.append([x, y])
    return np.array(points)


def _initial_points_random():
    """Random initial configuration."""
    rng = random.Random(N)
    return np.array([[rng.random() * 2 - 1, rng.random() * 2 - 1] for _ in range(N)])


def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_points_regular_polygon(),
        _initial_points_concentric_rings(),
        _initial_points_grid(),
        _initial_points_random(),
    ]
    
    best_points = None
    best_score = -np.inf
    
    for init in initial_configs:
        if time.time() > deadline - 5:
            break
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                _objective,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < -best_score:
                points = result.x.reshape(N, 2)
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, use the best initial configuration
    if best_points is None:
        best_points = initial_configs[0]
        best_score = _score(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
