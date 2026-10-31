"""Heilbronn triangles: maximize the smallest triangle area among 11 points in a unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle with area 1 (side length 1)
# Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
# Area = sqrt(3)/4 ≈ 0.4330127

def point_in_triangle(p):
    """Check if point p is inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2)"""
    x, y = p
    if y < -1e-9:
        return False
    if y > np.sqrt(3) / 2 + 1e-9:
        return False
    # Left edge: x >= y / sqrt(3)
    # Right edge: x <= 1 - y / sqrt(3)
    if x < y / np.sqrt(3) - 1e-9:
        return False
    if x > 1 - y / np.sqrt(3) + 1e-9:
        return False
    return True

def compute_min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle = 0.5 * |cross product|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective function to minimize (negative of min area, with penalty for out-of-bounds)."""
    points = x.reshape(-1, 2)
    # Check if all points are inside the triangle
    for p in points:
        if not point_in_triangle(p):
            # Penalty for being outside
            penalty = 0.0
            x, y = p
            # Distance to each edge
            if y < 0:
                penalty += -y
            if y > np.sqrt(3) / 2:
                penalty += y - np.sqrt(3) / 2
            if x < y / np.sqrt(3):
                penalty += y / np.sqrt(3) - x
            if x > 1 - y / np.sqrt(3):
                penalty += x - (1 - y / np.sqrt(3))
            if penalty > 1e-6:
                return 1.0 + penalty
    min_area = compute_min_triangle_area(points)
    return -min_area

def soft_min_area(points, temperature=0.01):
    """Soft minimum of triangle areas using log-sum-exp for differentiability."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log-sum-exp(-areas/T)
    min_val = np.min(areas)
    log_sum_exp = min_val + temperature * np.log(np.sum(np.exp(-(areas - min_val) / temperature)))
    return -log_sum_exp

def soft_objective(x, temperature=0.01):
    """Differentiable soft objective."""
    points = x.reshape(-1, 2)
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += (-y) ** 2
        if y > np.sqrt(3) / 2:
            penalty += (y - np.sqrt(3) / 2) ** 2
        if x < y / np.sqrt(3):
            penalty += (y / np.sqrt(3) - x) ** 2
        if x > 1 - y / np.sqrt(3):
            penalty += (x - (1 - y / np.sqrt(3))) ** 2
    if penalty > 1e-4:
        return 1.0 + penalty
    return -soft_min_area(points, temperature)

def generate_initial_points(rng, n=N):
    """Generate diverse initial points inside the triangle."""
    points = []
    # Include vertices and edge midpoints for diversity
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    edge_mids = np.array([[0.5, 0], [0.75, np.sqrt(3) / 4], [0.25, np.sqrt(3) / 4]])
    
    # Start with some structured points
    candidates = list(vertices) + list(edge_mids)
    
    # Add random points
    while len(candidates) < n:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            x = u + 0.5 * v
            y = np.sqrt(3) / 2 * v
            candidates.append([x, y])
    
    # Shuffle and take n
    rng.shuffle(candidates)
    points = np.array(candidates[:n])
    return points

def coordinate_descent(points, max_iter=500, tol=1e-8):
    """Refine points using coordinate descent on the soft objective."""
    x = points.flatten()
    best_val = soft_objective(x, 0.005)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            for dim in range(2):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001, 0.02, -0.02]:
                    x_trial = x.copy()
                    x_trial[i * 2 + dim] += delta
                    points_trial = x_trial.reshape(-1, 2)
                    # Check bounds
                    valid = True
                    for p in points_trial:
                        if not point_in_triangle(p):
                            valid = False
                            break
                    if not valid:
                        continue
                    val = soft_objective(x_trial, 0.005)
                    if val < best_val - 1e-10:
                        x = x_trial
                        best_val = val
                        improved = True
        if not improved:
            break
    
    return x.reshape(-1, 2)

def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Multiple restarts with diverse initializations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate initial points
        if restart == 0:
            # Structured initialization
            points = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2],
                [0.5, 0.0],
                [0.75, np.sqrt(3) / 4],
                [0.25, np.sqrt(3) / 4],
                [0.5, np.sqrt(3) / 6],
                [1.0 / 3, np.sqrt(3) / 6],
                [2.0 / 3, np.sqrt(3) / 6],
                [0.5, np.sqrt(3) / 3],
                [0.5, np.sqrt(3) * 2 / 3],
            ])
        else:
            points = generate_initial_points(rng)
        
        # Refine using coordinate descent
        refined = coordinate_descent(points, max_iter=200)
        
        # Compute actual min area
        min_area = compute_min_triangle_area(refined)
        
        if min_area > best_score:
            best_score = min_area
            best_points = refined.copy()
    
    # Final polish: try to improve with smaller steps
    if best_points is not None and time.time() < deadline - 10:
        x = best_points.flatten()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for dim in range(2):
                    for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                        x_trial = x.copy()
                        x_trial[i * 2 + dim] += delta
                        points_trial = x_trial.reshape(-1, 2)
                        valid = all(point_in_triangle(p) for p in points_trial)
                        if not valid:
                            continue
                        val = compute_min_triangle_area(points_trial)
                        if val > best_score + 1e-10:
                            x = x_trial
                            best_score = val
                            best_points = points_trial.copy()
                            improved = True
            if not improved:
                break
    
    if best_points is None:
        # Fallback: return a reasonable default
        best_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
            [0.5, 0.0],
            [0.75, np.sqrt(3) / 4],
            [0.25, np.sqrt(3) / 4],
            [0.5, np.sqrt(3) / 6],
            [1.0 / 3, np.sqrt(3) / 6],
            [2.0 / 3, np.sqrt(3) / 6],
            [0.5, np.sqrt(3) / 3],
            [0.5, np.sqrt(3) * 2 / 3],
        ])
    
    return best_points
