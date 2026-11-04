"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                cross = dx1 * dy2 - dy1 * dx2
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin with given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])

def _local_optimize(points, max_iter=500, step_size=0.05, time_limit=120.0):
    """Iteratively optimize points to improve the score."""
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    
    n = len(current_points)
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 5:
                    break
            
            best_neighbor = current_points.copy()
            best_neighbor_score = current_score
            
            # Try random perturbations
            for _ in range(10):
                direction = rng.normal(size=2)
                direction /= (np.linalg.norm(direction) + 1e-12)
                new_point = current_points[i] + step_size * direction
                trial_points = current_points.copy()
                trial_points[i] = new_point
                
                trial_score = _score(trial_points)
                if trial_score > best_neighbor_score + 1e-12:
                    best_neighbor_score = trial_score
                    best_neighbor = trial_points.copy()
            
            if best_neighbor_score > current_score + 1e-12:
                current_points = best_neighbor
                current_score = best_neighbor_score
                improved = True
        
        if not improved:
            # Try smaller step sizes
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return current_points, current_score

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin before 150s kill
    
    # Start with a regular 14-gon
    points = _regular_polygon(N, radius=1.0)
    initial_score = _score(points)
    
    # Local optimization
    optimized_points, optimized_score = _local_optimize(
        points, max_iter=300, step_size=0.05, time_limit=time_limit
    )
    
    if time.time() - start_time > time_limit - 5:
        return points  # Return initial if we're running out of time
    
    # Try a few different initial configurations
    best_points = optimized_points
    best_score = max(optimized_score, initial_score)
    
    if time.time() - start_time < time_limit - 10:
        # Try with slightly different initial radii or perturbations
        for radius in [0.9, 1.1]:
            if time.time() - start_time > time_limit - 10:
                break
            p = _regular_polygon(N, radius=radius)
            opt_p, opt_s = _local_optimize(p, max_iter=100, step_size=0.05, time_limit=time_limit)
            if opt_s > best_score:
                best_points = opt_p
                best_score = opt_s
    
    return best_points
# EVOLVE-BLOCK-END
