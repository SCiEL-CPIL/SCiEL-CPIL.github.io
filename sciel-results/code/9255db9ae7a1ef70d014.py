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

def _local_optimize(points, max_iter=500):
    """
    Perform local optimization using Nelder-Mead.
    Returns (optimized_points, score).
    """
    x0 = np.array(points).flatten()
    
    def objective(x_flat):
        pts = np.array(x_flat).reshape(-1, 2)
        pts = np.array([_clamp_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts)

    # Nelder-Mead is non-gradient and robust for this non-smooth objective
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        opt_points = np.array(result.x).reshape(-1, 2)
        opt_points = np.array([_clamp_to_triangle(p) for p in opt_points])
        return opt_points, _evaluate(opt_points)
    except Exception:
        return points, _evaluate(points)

def _generate_initial_points():
    """Generate a list of diverse initial configurations."""
    h = np.sqrt(3) / 2
    inits = []
    
    # 1. Symmetric configuration
    sym_points = np.array([
        [0.0, 0.0],      # vertex 1
        [1.0, 0.0],      # vertex 2
        [0.5, h],         # vertex 3
        [0.5, h/3],       # centroid
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4],
    ])
    inits.append(sym_points)
    
    # 2. Random configurations (seeded for reproducibility but diverse)
    rng = np.random.default_rng(42)
    for i in range(20):
        pts = rng.random((N, 2))
        # Scale to fit roughly in triangle, then clamp
        pts[:, 0] = pts[:, 0]
        pts[:, 1] = pts[:, 1] * h
        pts = np.array([_clamp_to_triangle(p) for p in pts])
        inits.append(pts)
        
    # 3. Grid-like or structured perturbations
    # Vertices + Centroid + 7 points on a hexagonal grid within the triangle
    # This is a heuristic for good packing
    hex_points = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, h/3],
        [0.25, h/6], [0.75, h/6], [0.5, h/2],
        [0.167, h/3], [0.833, h/3], [0.5, h/6],
        [0.333, h/6], [0.667, h/6] # Adjusting to get 11 distinct
    ]
    # Ensure we have exactly 11 points, remove duplicates if any
    unique_hex = []
    for p in hex_points:
        if not any(np.allclose(p, q) for q in unique_hex):
            unique_hex.append(p)
        if len(unique_hex) == N:
            break
    if len(unique_hex) == N:
        inits.append(np.array(unique_hex))

    return inits

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    # Get initial configurations
    inits = _generate_initial_points()
    
    best_points = inits[0]
    best_score = _evaluate(best_points)
    
    # Iterate through initial configurations and perform local optimization
    # We prioritize the symmetric one first, then randoms
    # To save time, we limit the number of full optimizations if time is running out
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # If we are getting close to deadline, reduce max_iter
        time_left = deadline - time.time()
        if time_left < 10:
            max_iter = 50
        elif time_left < 30:
            max_iter = 200
        else:
            max_iter = 500
            
        try:
            opt_points, opt_score = _local_optimize(init_pts, max_iter=max_iter)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception:
            continue
            
        # If we have a very good score, we might stop early or refine
        # The seed score is ~0.0011. A good solution is ~0.002-0.003.
        # We continue to ensure we explore the basin.
        
    # Final polish on the best found
    if time.time() < deadline - 5:
        try:
            opt_points, opt_score = _local_optimize(best_points, max_iter=1000)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
        except Exception:
            pass

    # Ensure all points are strictly inside/on boundary
    best_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return best_points

# EVOLVE-BLOCK-END
