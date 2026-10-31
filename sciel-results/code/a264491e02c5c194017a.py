"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    # Constraints:
    # y >= 0
    # y <= h (max height)
    # y <= sqrt(3) * x  (left side, slope = h/0.5 = sqrt(3))
    # y <= sqrt(3) * (1 - x)  (right side)
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    # From y <= sqrt(3)*x => x >= y/sqrt(3)
    x_min_left = y / np.sqrt(3)
    # From y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _evaluate(points):
    """Evaluate the minimum triangle area for a set of points."""
    return _min_triangle_area(points)

def _local_optimize(points, max_iter=2000):
    """
    Perform local optimization using Nelder-Mead.
    Returns (optimized_points, score).
    """
    x0 = np.array(points).flatten()
    
    def objective(x_flat):
        pts = np.array(x_flat).reshape(-1, 2)
        pts = np.array([_clamp_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts)
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    opt_points = np.array(result.x).reshape(-1, 2)
    opt_points = np.array([_clamp_to_triangle(p) for p in opt_points])
    return opt_points, _evaluate(opt_points)

def _generate_initial_points(seed=None):
    """Generate diverse initial point configurations."""
    h = np.sqrt(3) / 2
    
    if seed is not None:
        np.random.seed(seed)
    
    configs = []
    
    # Configuration 1: Vertices + centroid + symmetric points
    config1 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4],
    ]
    configs.append(config1)
    
    # Configuration 2: More spread out
    config2 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.2, h/5],
        [0.8, h/5],
        [0.5, h/2],
        [0.1, h/10],
        [0.9, h/10],
        [0.35, h/4],
        [0.65, h/4],
    ]
    configs.append(config2)
    
    # Configuration 3: Random symmetric
    config3 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3 + 0.03 * np.random.randn()],
        [0.25 + 0.03 * np.random.randn(), h/6 + 0.02 * np.random.randn()],
        [0.75 + 0.03 * np.random.randn(), h/6 + 0.02 * np.random.randn()],
        [0.5 + 0.03 * np.random.randn(), h/2 + 0.02 * np.random.randn()],
        [0.125 + 0.03 * np.random.randn(), h/12 + 0.02 * np.random.randn()],
        [0.875 + 0.03 * np.random.randn(), h/12 + 0.02 * np.random.randn()],
        [0.375 + 0.03 * np.random.randn(), h/4 + 0.02 * np.random.randn()],
        [0.625 + 0.03 * np.random.randn(), h/4 + 0.02 * np.random.randn()],
    ]
    configs.append(config3)
    
    # Configuration 4: Random
    config4 = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = h * v
        config4.append([x, y])
    configs.append(config4)
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    h = np.sqrt(3) / 2
    
    # Generate initial configurations
    initial_configs = _generate_initial_points(seed=42)
    
    best_points = np.array(initial_configs[0])
    best_score = _evaluate(best_points)
    
    # Optimize each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        points = np.array([_clamp_to_triangle(p) for p in config])
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points
            best_score = current_score
        
        # Local optimization
        if time.time() < deadline - 20:
            opt_points, opt_score = _local_optimize(points, max_iter=1500)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Try additional random restarts if time permits
    np.random.seed(123)
    for restart in range(20):
        if time.time() > deadline - 25:
            break
        
        # Generate random points
        points = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = h * v
            points.append([x, y])
        
        # Also try structured random
        if restart % 2 == 0:
            base = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.5, h/3 + 0.05 * np.random.randn()],
                [0.25 + 0.05 * np.random.randn(), h/6 + 0.03 * np.random.randn()],
                [0.75 + 0.05 * np.random.randn(), h/6 + 0.03 * np.random.randn()],
                [0.5 + 0.05 * np.random.randn(), h/2 + 0.03 * np.random.randn()],
                [0.125 + 0.05 * np.random.randn(), h/12 + 0.03 * np.random.randn()],
                [0.875 + 0.05 * np.random.randn(), h/12 + 0.03 * np.random.randn()],
                [0.375 + 0.05 * np.random.randn(), h/4 + 0.03 * np.random.randn()],
                [0.625 + 0.05 * np.random.randn(), h/4 + 0.03 * np.random.randn()],
            ]
            points = base
        
        points = np.array([_clamp_to_triangle(p) for p in points])
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points
            best_score = current_score
        
        if time.time() < deadline - 15:
            opt_points, opt_score = _local_optimize(points, max_iter=1000)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Final check: ensure all points are inside the triangle
    best_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return best_points

# EVOLVE-BLOCK-END
