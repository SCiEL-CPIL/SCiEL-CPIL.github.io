"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
TRIANGLE_AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    min_area = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    # Barycentric coordinates
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _project_to_triangle(p):
    """Project point p to the nearest point in the equilateral triangle."""
    if _in_triangle(p):
        return p.copy()
    
    # Check distance to each edge and project to closest edge
    best = p.copy()
    best_dist = float('inf')
    
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    
    return best


def _objective(x):
    """Negative of minimum triangle area (for minimization)."""
    points = x.reshape(-1, 2)
    # Project points to triangle to handle constraint violations softly
    # But for the objective, we just compute min area
    min_area = _min_triangle_area(points)
    return -min_area


def _constraint_penalty(x, weight=100.0):
    """Penalty for points outside the triangle."""
    points = x.reshape(-1, 2)
    penalty = 0.0
    for i in range(len(points)):
        p = points[i]
        if not _in_triangle(p, eps=1e-6):
            # Distance to nearest point in triangle
            proj = _project_to_triangle(p)
            dist = np.linalg.norm(p - proj)
            penalty += weight * dist
    return penalty


def _objective_with_penalty(x, weight=100.0):
    """Objective with penalty for constraint violation."""
    points = x.reshape(-1, 2)
    min_area = _min_triangle_area(points)
    penalty = 0.0
    for i in range(len(points)):
        p = points[i]
        if not _in_triangle(p, eps=1e-6):
            proj = _project_to_triangle(p)
            dist = np.linalg.norm(p - proj)
            penalty += weight * dist
    return -min_area + penalty


def _generate_initial_points(n, seed):
    """Generate n random points inside the triangle."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < n:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = u * V1 + v * V2
            # Add small jitter to avoid exact vertices
            p = p + rng.normal(0, 0.01, 2)
            if _in_triangle(p):
                points.append(p)
    return np.array(points)


def _optimize_from_start(points, max_iter=500, tol=1e-8):
    """Optimize starting from given points using scipy minimize."""
    x0 = points.flatten()
    
    # Use SLSQP with bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x coordinate
        bounds.append((0.0, np.sqrt(3) / 2.0))  # y coordinate
    
    result = minimize(
        _objective_with_penalty,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
    )
    
    if result.success:
        return result.x.reshape(-1, 2), result.fun
    else:
        return x0.reshape(-1, 2), _objective_with_penalty(x0)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Generate multiple random initializations
    n_starts = 20
    
    for seed in range(n_starts):
        if time.time() > deadline:
            break
        
        points = _generate_initial_points(N, seed)
        optimized_points, score = _optimize_from_start(points, max_iter=300, tol=1e-7)
        
        # Project to triangle to ensure validity
        valid_points = np.array([_project_to_triangle(p) for p in optimized_points])
        
        # Compute actual score
        min_area = _min_triangle_area(valid_points)
        normalized_score = min_area / TRIANGLE_AREA
        
        if normalized_score > best_score:
            best_score = normalized_score
            best_points = valid_points
        
        # Check if we should stop early
        if time.time() > deadline - 10.0:
            break
    
    # If we haven't found any valid points, use a fallback
    if best_points is None:
        best_points = _generate_initial_points(N, 0)
    
    return best_points
