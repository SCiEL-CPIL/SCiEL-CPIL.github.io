"""Heilbronn convex regions n=14: maximize minimum triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Sort by polar angle
    def angle_cmp(p):
        return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
    
    sorted_pts = sorted(points, key=angle_cmp)
    sorted_pts.insert(0, points[start])
    
    # Build convex hull using Graham scan
    hull = []
    for p in sorted_pts:
        while len(hull) >= 2:
            p1 = hull[-1]
            p2 = hull[-2]
            cross = (p1[0] - p2[0]) * (p[1] - p2[1]) - (p1[1] - p2[1]) * (p[0] - p2[0])
            if cross <= 1e-12:  # non-left turn
                hull.pop()
            else:
                break
        hull.append(p)
    
    if len(hull) < 3:
        return 0.0
    
    # Compute area using shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    
    return abs(area) / 2.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _initial_grid_config(n=N):
    """Generate an initial configuration using a grid-like pattern."""
    # Use a regular polygon as starting point
    angles = 2 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _random_config(n=N, seed=42):
    """Generate a random configuration inside unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2)

def _optimize_from(points, time_limit=100.0):
    """Optimize a configuration using scipy minimize."""
    n = len(points)
    flat = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    start_time = time.time()
    best_x = flat.copy()
    best_score = -objective(flat)
    
    # Use Nelder-Mead for robustness
    try:
        result = minimize(objective, flat, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        if result.fun < objective(best_x):
            best_x = result.x
            best_score = -result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_x.reshape(-1, 2), best_score, elapsed

def construct():
    """Main entry point: construct 14 points to maximize min triangle area / hull area."""
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Try multiple initial configurations
    configs = []
    
    # Regular polygon
    configs.append(_initial_grid_config(N))
    
    # Several random configs
    for seed in range(5):
        configs.append(_random_config(N, seed=seed))
    
    # Add a few structured configs
    # Square grid
    grid = []
    for i in range(4):
        for j in range(4):
            grid.append([i/3.0, j/3.0])
    configs.append(np.array(grid[:N]))
    
    # Hexagonal-like pattern
    hex_pts = []
    for i in range(4):
        for j in range(4):
            x = i/3.0 + (j % 2) * 0.08
            y = j/3.0
            hex_pts.append([x, y])
    configs.append(np.array(hex_pts[:N]))
    
    best_points = configs[0].copy()
    best_score = _score(configs[0])
    
    for cfg in configs:
        if time.time() > deadline - 30:
            break
        pts, score, elapsed = _optimize_from(cfg, time_limit=deadline - time.time() - 5)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        flat = best_points.flatten()
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        try:
            result = minimize(objective, flat, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True})
            if result.fun < objective(flat):
                best_points = result.x.reshape(-1, 2)
                best_score = -result.fun
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
