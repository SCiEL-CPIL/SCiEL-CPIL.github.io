"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
# Area = sqrt(3)/4
_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, np.sqrt(3) / 2])
_TRIG_AREA = np.sqrt(3) / 4

def _in_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, _A, _B)
    d2 = sign(p, _B, _C)
    d3 = sign(p, _C, _A)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def _project_to_triangle(p):
    """Project point p onto the triangle (closest point)."""
    if _in_triangle(p):
        return p
    
    # Check edges
    def dist_to_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return np.linalg.norm(p - closest), closest
    
    best_dist = float('inf')
    best_point = None
    
    for a, b in [(_A, _B), (_B, _C), (_C, _A)]:
        d, cp = dist_to_segment(p, a, b)
        if d < best_dist:
            best_dist = d
            best_point = cp
    
    return best_point

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    return min_area

def _score(points):
    """Score = min_triangle_area / triangle_area."""
    return _min_triangle_area(points) / _TRIG_AREA

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial configuration: grid-like placement
    # Use barycentric coordinates for better distribution
    best_points = None
    best_score = -1
    
    # Start with a reasonable initial configuration
    # Place points on a hexagonal grid pattern
    def initial_config():
        pts = []
        # Corner points (slightly inset)
        eps = 0.05
        pts.append(_A + eps * (_B - _A) + eps * (_C - _A))
        pts.append(_B + eps * (_A - _B) + eps * (_C - _B))
        pts.append(_C + eps * (_A - _C) + eps * (_B - _C))
        
        # Midpoints of edges (inset)
        mid_ab = 0.5 * (_A + _B)
        mid_bc = 0.5 * (_B + _C)
        mid_ca = 0.5 * (_C + _A)
        pts.append(mid_ab)
        pts.append(mid_bc)
        pts.append(mid_ca)
        
        # Interior points
        centroid = (_A + _B + _C) / 3
        for angle in range(0, 360, 60):
            rad = np.radians(angle)
            offset = 0.15 * np.array([np.cos(rad), np.sin(rad)])
            p = centroid + offset
            if _in_triangle(p):
                pts.append(p)
            else:
                pts.append(centroid)
        
        # Ensure we have exactly 11 points
        while len(pts) < 11:
            p = centroid + 0.05 * np.random.randn(2)
            if _in_triangle(p):
                pts.append(p)
            else:
                pts.append(centroid)
        
        return np.array(pts[:11])
    
    # Local optimization using gradient-free method
    def local_optimize(points, max_iter=5000, step_size=0.01):
        points = points.copy()
        current_score = _score(points)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(N):
                # Try moving point i in random directions
                for _ in range(20):
                    direction = np.random.randn(2)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    new_point = points[i] + step_size * direction
                    
                    if not _in_triangle(new_point):
                        new_point = _project_to_triangle(new_point)
                    
                    old_point = points[i].copy()
                    points[i] = new_point
                    new_score = _score(points)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        points[i] = old_point
                
                if not improved:
                    # Reduce step size
                    step_size *= 0.95
                    if step_size < 1e-6:
                        break
            
            if step_size < 1e-6:
                break
        
        return points, current_score
    
    # Multi-start local optimization
    np.random.seed(42)
    
    # First, try the initial configuration
    points = initial_config()
    optimized, score = local_optimize(points, max_iter=3000, step_size=0.05)
    if score > best_score:
        best_score = score
        best_points = optimized.copy()
    
    # Multiple random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial configuration
        pts = []
        for _ in range(N):
            while True:
                u, v = np.random.random(), np.random.random()
                if u + v < 1:
                    p = u * _B + v * _C
                    pts.append(p)
                    break
        
        points = np.array(pts)
        optimized, score = local_optimize(points, max_iter=2000, step_size=0.03)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    # Final polish with smaller steps
    if best_points is not None and time.time() < deadline - 10:
        polished, final_score = local_optimize(best_points, max_iter=5000, step_size=0.005)
        if final_score > best_score:
            best_score = final_score
            best_points = polished
    
    return best_points

# EVOLVE-BLOCK-END
