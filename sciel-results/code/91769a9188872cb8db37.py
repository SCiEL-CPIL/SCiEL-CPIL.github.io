"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _negative_min_area(x):
    """Objective function: negative of the minimum triangle area."""
    points = x.reshape(N, 2)
    return -_min_triangle_area(points)

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    h = math.sqrt(3) / 2
    x, y = p
    # barycentric coordinates check
    # y >= 0
    # y <= h * x (left edge from (0,0) to (0.5,h))
    # y <= h * (1 - x) (right edge from (1,0) to (0.5,h))
    if y < -1e-9:
        return False
    if y > h * x + 1e-9:
        return False
    if y > h * (1 - x) + 1e-9:
        return False
    return True

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    h = math.sqrt(3) / 2
    clamped = points.copy()
    for i in range(len(clamped)):
        x, y = clamped[i]
        # Simple projection: if outside, move toward centroid
        centroid = np.array([0.5, h / 3])
        if not _is_inside_triangle(clamped[i]):
            # Binary search for t such that point is inside
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = (lo + hi) / 2
                pt = clamped[i] * (1 - mid) + centroid * mid
                if _is_inside_triangle(pt):
                    hi = mid
                else:
                    lo = mid
            t = (lo + hi) / 2
            clamped[i] = clamped[i] * (1 - t) + centroid * t
    return clamped

def _refine(points):
    """Refine points using coordinate descent / local optimization."""
    points = points.copy()
    h = math.sqrt(3) / 2
    best = _min_triangle_area(points)
    
    for iteration in range(50):
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    new_pt = points[i] + np.array([dx, dy])
                    if _is_inside_triangle(new_pt):
                        old_pt = points[i].copy()
                        points[i] = new_pt
                        new_best = _min_triangle_area(points)
                        if new_best > best + 1e-10:
                            best = new_best
                            improved = True
                        else:
                            points[i] = old_pt
        if not improved:
            break
    return points, best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave margin
    
    h = math.sqrt(3) / 2
    centroid = np.array([0.5, h / 3])
    
    best_points = None
    best_value = 0.0
    
    # Strategy: Use differential evolution with multiple restarts
    # Bounds for each coordinate
    bounds = [(0, 1), (0, h)] * N
    
    # Initial structured guesses
    structured_inits = []
    
    # Init 1: Grid-like distribution
    grid_pts = []
    for i in range(4):
        for j in range(3):
            x = (i + 0.5) / 4
            y = (j + 0.5) / 3 * h * (1 - 2 * abs(x - 0.5))  # Adjust for triangle shape
            if 0 <= x <= 1 and 0 <= y <= h and _is_inside_triangle([x, y]):
                grid_pts.append([x, y])
    # Add centroid if not already there
    if len(grid_pts) < N:
        grid_pts.append(centroid.tolist())
    # Pad if needed
    while len(grid_pts) < N:
        grid_pts.append(centroid.tolist())
    structured_inits.append(np.array(grid_pts[:N]).flatten())
    
    # Init 2: Vertices + edge midpoints + interior
    init2 = []
    vertices = [[0, 0], [1, 0], [0.5, h]]
    edge_mids = [[0.5, 0], [0.75, h/2], [0.25, h/2]]
    interior = [centroid.tolist()]
    # Add more interior points
    interior.append([0.3, h/4])
    interior.append([0.7, h/4])
    interior.append([0.5, h/2])
    interior.append([0.4, h/3])
    init2 = vertices + edge_mids + interior
    while len(init2) < N:
        init2.append(centroid.tolist())
    structured_inits.append(np.array(init2[:N]).flatten())
    
    # Init 3: Random but seeded
    np.random.seed(42)
    init3 = []
    attempts = 0
    while len(init3) < N and attempts < 1000:
        u, v = np.random.random(2)
        if u + v < 1:
            x = u + 0.5 * v
            y = h * v
            init3.append([x, y])
        attempts += 1
    while len(init3) < N:
        init3.append(centroid.tolist())
    structured_inits.append(np.array(init3[:N]).flatten())
    
    # Refine each initial guess
    for init in structured_inits:
        if time.time() > deadline:
            break
        points = init.reshape(N, 2)
        points = _clamp_to_triangle(points)
        refined_pts, val = _refine(points)
        if val > best_value:
            best_value = val
            best_points = refined_pts
    
    # Now run differential evolution with remaining time
    if time.time() < deadline - 30:
        # Run DE
        def objective(x):
            points = _clamp_to_triangle(x.reshape(N, 2))
            return -_min_triangle_area(points)
        
        # Use the best initial point as starting point
        if best_points is not None:
            x0 = best_points.flatten()
        else:
            x0 = structured_inits[0]
        
        # Run DE with limited time
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=100, popsize=15,
                seed=42, tol=1e-8,
                init='sobol',
                polish=False
            )
            if time.time() < deadline - 10:
                de_pts = _clamp_to_triangle(result.x.reshape(N, 2))
                de_val = _min_triangle_area(de_pts)
                if de_val > best_value:
                    best_value = de_val
                    best_points = de_pts
        except Exception:
            pass
    
    # Final refinement with remaining time
    if best_points is not None and time.time() < deadline - 5:
        refined_pts, val = _refine(best_points)
        if val > best_value:
            best_value = val
            best_points = refined_pts
    
    if best_points is None:
        # Fallback: return centroid-based points
        best_points = np.tile(centroid, (N, 1))
    
    # Ensure all points are strictly inside
    best_points = _clamp_to_triangle(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
