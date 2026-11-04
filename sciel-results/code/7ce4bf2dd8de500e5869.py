"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
S3 = np.sqrt(3.0)
TRI_AREA = S3 / 4.0  # Area of the equilateral triangle with side length 1

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            for k in range(j + 1, n):
                pk = points[k]
                dx2 = pk[0] - pi[0]
                dy2 = pk[1] - pi[1]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_a:
                    min_a = area
                    if min_a < 1e-15:
                        return 0.0
    return min_a

def _in_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-12 or x < -1e-12:
        return False
    if y > S3 / 2 + 1e-12:
        return False
    # Check right side: line from (1,0) to (0.5, sqrt(3)/2)
    # The line equation: y = -sqrt(3)(x - 1) => sqrt(3)*x + y = sqrt(3)
    if x > 0.5:
        if y > -S3 * (x - 1) + 1e-12:
            return False
    else:
        # Left side: line from (0,0) to (0.5, sqrt(3)/2)
        # y = sqrt(3)*x
        if y > S3 * x + 1e-12:
            return False
    return True

def _project_to_triangle(p):
    """Project point p onto the equilateral triangle (closest point)."""
    x, y = p
    # Simple approach: if inside, return as is
    if _in_triangle(p):
        return np.array([x, y])
    
    # Otherwise, find closest point on boundary
    # Edges:
    # 1. Bottom: (0,0) to (1,0)
    # 2. Right: (1,0) to (0.5, sqrt(3)/2)
    # 3. Left: (0,0) to (0.5, sqrt(3)/2)
    
    def dist_to_segment(p, a, b):
        a = np.array(a)
        b = np.array(b)
        p = np.array(p)
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj), proj
    
    d1, proj1 = dist_to_segment(p, [0, 0], [1, 0])
    d2, proj2 = dist_to_segment(p, [1, 0], [0.5, S3/2])
    d3, proj3 = dist_to_segment(p, [0, 0], [0.5, S3/2])
    
    min_d = min(d1, d2, d3)
    if min_d == d1:
        return proj1
    elif min_d == d2:
        return proj2
    else:
        return proj3

def _objective(params):
    """Negative of minimum triangle area (to be minimized)."""
    points = np.array(params).reshape(N, 2)
    # Project to triangle to handle boundary
    for i in range(N):
        if not _in_triangle(points[i]):
            points[i] = _project_to_triangle(points[i])
    min_a = _min_triangle_area(points)
    return -min_a

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_min_area = 0.0
    
    # Grid of starting configurations
    np.random.seed(42)
    
    def evaluate_and_update(points):
        nonlocal best_points, best_min_area
        # Project to triangle
        for i in range(N):
            if not _in_triangle(points[i]):
                points[i] = _project_to_triangle(points[i])
        min_a = _min_triangle_area(points)
        if min_a > best_min_area:
            best_min_area = min_a
            best_points = points.copy()
        return min_a
    
    # Try several random initial configurations
    n_attempts = 0
    max_attempts = 50
    
    while n_attempts < max_attempts:
        if time.time() > deadline - 20.0:
            break
        
        # Generate random initial points
        points = np.zeros((N, 2))
        for i in range(N):
            u, v = np.random.random(), np.random.random()
            while u + v >= 1:
                u, v = np.random.random(), np.random.random()
            points[i] = [u + 0.5 * v, S3 / 2 * v]
        
        # Add vertices to ensure good coverage
        points[0] = [0.05, 0.05]
        points[1] = [0.95, 0.05]
        points[2] = [0.5, S3/2 - 0.05]
        
        evaluate_and_update(points)
        
        # Local optimization
        if time.time() > deadline - 15.0:
            break
        
        x0 = points.flatten()
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() > deadline - 5.0:
                break
            optimized_points = res.x.reshape(N, 2)
            evaluate_and_update(optimized_points)
        except Exception:
            pass
        
        n_attempts += 1
    
    if best_points is None:
        # Fallback: simple grid
        best_points = np.zeros((N, 2))
        for i in range(N):
            u, v = np.random.random(), np.random.random()
            while u + v >= 1:
                u, v = np.random.random(), np.random.random()
            best_points[i] = [u + 0.5 * v, S3 / 2 * v]
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0 and best_points is not None:
        x0 = best_points.flatten()
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            polished_points = res.x.reshape(N, 2)
            for i in range(N):
                if not _in_triangle(polished_points[i]):
                    polished_points[i] = _project_to_triangle(polished_points[i])
            min_a = _min_triangle_area(polished_points)
            if min_a > best_min_area:
                best_points = polished_points
                best_min_area = min_a
        except Exception:
            pass
    
    return np.array(best_points)
