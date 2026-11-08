"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    # Precompute cross products for all pairs
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            p_ji = points[j] - pi
            for k in range(j + 1, n):
                p_ki = points[k] - pi
                area = 0.5 * abs(p_ji[0] * p_ki[1] - p_ji[1] * p_ki[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area from sorted hull points
        from scipy.spatial import ConvexHull as CH
        try:
            hull = CH(points)
            return hull.volume
        except:
            return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _normalize_to_unit_area(points):
    """Scale points so their convex hull has unit area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale


def _initial_regular_polygon(n, radius=1.0):
    """Create n points on a regular polygon."""
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _initial_fibonacci_sphere(n):
    """Create n points using a Fibonacci-like distribution in 2D."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        angle = 2 * np.pi * i / golden_ratio
        radius = np.sqrt((i + 0.5) / n)
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(points)


def _coordinate_descent(points, max_iter=500, tol=1e-8):
    """Refine points using coordinate descent to maximize min triangle area / hull area."""
    points = points.copy()
    best_score = _score(points)
    current_score = best_score
    
    for iteration in range(max_iter):
        improved = False
        for i in range(len(points)):
            orig_point = points[i].copy()
            best_local_score = current_score
            best_local_point = orig_point.copy()
            
            # Try small perturbations in x and y directions
            for axis in [0, 1]:
                for step in [0.01, 0.05, 0.1, 0.2]:
                    for direction in [-1, 1]:
                        test_point = points.copy()
                        test_point[i, axis] += direction * step
                        test_score = _score(test_point)
                        if test_score > best_local_score:
                            best_local_score = test_score
                            best_local_point = test_point[i].copy()
            
            if np.any(best_local_point != orig_point):
                points[i] = best_local_point
                current_score = best_local_score
                if current_score > best_score:
                    best_score = current_score
                    improved = True
        
        if not improved:
            break
        if current_score - best_score < tol:
            break
    
    return points, best_score


def _powell_like_optimization(points, max_iter=100):
    """Simple Powell-like optimization for better local search."""
    points = points.copy()
    best_score = _score(points)
    current_score = best_score
    
    for iteration in range(max_iter):
        improved = False
        for i in range(len(points)):
            orig_point = points[i].copy()
            best_local_score = current_score
            best_local_point = orig_point.copy()
            
            # Adaptive step size
            step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]
            for axis in [0, 1]:
                for step in step_sizes:
                    for direction in [-1, 1]:
                        test_point = points.copy()
                        test_point[i, axis] += direction * step
                        test_score = _score(test_point)
                        if test_score > best_local_score:
                            best_local_score = test_score
                            best_local_point = test_point[i].copy()
            
            if np.any(best_local_point != orig_point):
                points[i] = best_local_point
                current_score = best_local_score
                if current_score > best_score:
                    best_score = current_score
                    improved = True
        
        if not improved:
            break
    
    return points, best_score


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_points = None
    best_score = 0.0
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_initial_regular_polygon(N))
    
    # 2. Fibonacci distribution
    initializations.append(_initial_fibonacci_sphere(N))
    
    # 3. Random points with different seeds
    np.random.seed(42)
    initializations.append(np.random.random((N, 2)) * 2 - 1)
    
    np.random.seed(123)
    initializations.append(np.random.random((N, 2)) * 2 - 1)
    
    np.random.seed(777)
    initializations.append(np.random.random((N, 2)) * 2 - 1)
    
    # 4. Points on a circle with some jitter
    angles = 2 * np.pi * np.arange(N) / N
    radii = 1.0 + 0.1 * np.sin(3 * angles)
    initializations.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # 5. Grid-like pattern
    grid_points = []
    for i in range(4):
        for j in range(4):
            if len(grid_points) < N:
                grid_points.append([i / 3.0 - 0.5, j / 3.0 - 0.5])
    if len(grid_points) == N:
        initializations.append(np.array(grid_points))
    
    # Optimize each initialization
    for init_points in initializations:
        if time.time() > deadline - 20.0:
            break
        
        # Normalize to unit area first
        normalized = _normalize_to_unit_area(init_points)
        
        # Coordinate descent refinement
        refined, score = _coordinate_descent(normalized, max_iter=200)
        if score > best_score:
            best_score = score
            best_points = refined
        
        # Further refinement with Powell-like
        if time.time() > deadline - 10.0:
            break
        refined2, score2 = _powell_like_optimization(refined, max_iter=50)
        if score2 > best_score:
            best_score = score2
            best_points = refined2
    
    # If we still have time, do more random restarts
    if best_points is not None and time.time() < deadline - 30.0:
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            np.random.seed(seed * 137 + 1)
            random_points = np.random.random((N, 2)) * 2 - 1
            normalized = _normalize_to_unit_area(random_points)
            refined, score = _coordinate_descent(normalized, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = refined
    
    # Final normalization
    if best_points is not None:
        best_points = _normalize_to_unit_area(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
