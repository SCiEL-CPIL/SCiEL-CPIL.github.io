"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
S3 = np.sqrt(3.0)
# Equilateral triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])
AREA = S3 / 4.0  # area of the equilateral triangle with side 1

# Deadline: must finish within 120s, use 110s as safety margin
DEADLINE = 110.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                if area < min_area:
                    min_area = area
    return min_area


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1.0 + 1e-9:
        return False
    # Check below the line from V0 to V2: y <= sqrt(3)*x
    if y > S3 * x + 1e-9:
        return False
    # Check below the line from V1 to V2: y <= -sqrt(3)*(x-1)
    if y > S3 * (1.0 - x) + 1e-9:
        return False
    return True


def _random_point_in_triangle(rng):
    """Generate a random point uniformly inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    # Barycentric coordinates: (1-u-v, u, v) mapped to triangle
    p = (1.0 - u - v) * V0 + u * V1 + v * V2
    return p


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle."""
    points = []
    for _ in range(n):
        points.append(_random_point_in_triangle(rng))
    return np.array(points)


def _local_optimize(points, rng, max_iters=200, step_size=0.05):
    """Local optimization: iteratively move the point with smallest contribution to improve min area."""
    points = points.copy()
    current_min = _min_triangle_area(points)
    
    for iteration in range(max_iters):
        # Find the triangle with minimum area
        n = len(points)
        min_area = float('inf')
        min_tri = (0, 1, 2)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                     - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                    if area < min_area:
                        min_area = area
                        min_tri = (i, j, k)
        
        if min_area >= current_min:
            break
        
        # Try moving each point in the minimum triangle to improve
        improved = False
        for idx in min_tri:
            orig = points[idx].copy()
            # Try several random perturbations
            for _ in range(20):
                step = step_size * rng.random()
                angle = rng.random() * 2 * np.pi
                new_p = orig + step * np.array([np.cos(angle), np.sin(angle)])
                if _is_inside_triangle(new_p):
                    points[idx] = new_p
                    new_min = _min_triangle_area(points)
                    if new_min > current_min:
                        current_min = new_min
                        improved = True
                        break
                    else:
                        points[idx] = orig
            if improved:
                break
        
        if not improved:
            break
    
    return points, current_min


def _simulated_annealing(points, rng, start_temp=0.1, end_temp=0.001, max_iters=500):
    """Simulated annealing to optimize the minimum triangle area."""
    points = points.copy()
    current_min = _min_triangle_area(points)
    best_points = points.copy()
    best_min = current_min
    
    for i in range(max_iters):
        temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
        
        # Pick a random point and perturb it
        idx = rng.randint(0, N - 1)
        orig = points[idx].copy()
        step = 0.02 * rng.random()
        angle = rng.random() * 2 * np.pi
        new_p = orig + step * np.array([np.cos(angle), np.sin(angle)])
        
        if _is_inside_triangle(new_p):
            points[idx] = new_p
            new_min = _min_triangle_area(points)
            delta = new_min - current_min
            if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
                current_min = new_min
                if new_min > best_min:
                    best_min = new_min
                    best_points = points.copy()
            else:
                points[idx] = orig
    
    return best_points, best_min


def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    
    # Start with a structured initial configuration
    # Place points at vertices, midpoints of edges, and some interior points
    rng = random.Random(42)
    
    # Initial structured configuration
    points = np.array([
        V0 + 0.05 * np.array([1.0, 1.0/S3]),  # Near vertex 0
        V1 + 0.05 * np.array([-1.0, 1.0/S3]),  # Near vertex 1
        V2 + 0.05 * np.array([0.0, -1.0]),     # Near vertex 2
        (V0 + V1) / 2.0,  # Midpoint of edge 0-1
        (V1 + V2) / 2.0,  # Midpoint of edge 1-2
        (V2 + V0) / 2.0,  # Midpoint of edge 2-0
        (V0 + V1 + V2) / 3.0,  # Centroid
    ])
    
    # Add remaining random points
    for _ in range(N - len(points)):
        points = np.vstack([points, _random_point_in_triangle(rng)])
    
    best_points = points.copy()
    best_score = _min_triangle_area(points)
    
    # Multiple restarts with local optimization
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate new random configuration
        points = _generate_random_points(rng)
        
        # Local optimization
        points, score = _local_optimize(points, rng, max_iters=100, step_size=0.03)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Try simulated annealing on the best so far
        if time.time() - start_time > DEADLINE - 10:
            break
        sa_points, sa_score = _simulated_annealing(best_points, rng, start_temp=0.05, end_temp=0.001, max_iters=200)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Final local polish
    if time.time() - start_time < DEADLINE - 2:
        best_points, final_score = _local_optimize(best_points, rng, max_iters=50, step_size=0.01)
        if final_score > best_score:
            best_score = final_score
            best_points = best_points.copy()
    
    return best_points
