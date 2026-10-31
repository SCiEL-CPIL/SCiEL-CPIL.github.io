"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the smallest triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
TIME_LIMIT = 110  # safety margin before 120s kill

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a, b, c):
    """Check if point p is inside or on the boundary of triangle abc using barycentric coordinates."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1 + 1e-9)

def project_to_triangle(p):
    """Project point p onto the triangle (0,0),(1,0),(0.5,sqrt(3)/2) if it's outside."""
    if point_in_triangle(p, A, B, C):
        return p
    # Simple projection: find the closest point on the triangle boundary
    # This is a simplified version; for robustness, we'll use a constrained optimization approach
    # For now, clamp to the bounding box and check
    # A more robust approach would be needed, but let's use a simple method
    # If outside, project to the nearest edge
    edges = [(A, B), (B, C), (C, A)]
    min_dist = float('inf')
    closest = p
    for e1, e2 in edges:
        # Project p onto line segment e1-e2
        d = e2 - e1
        if np.dot(d, d) == 0:
            continue
        t = np.dot(p - e1, d) / np.dot(d, d)
        t = np.clip(t, 0, 1)
        proj = e1 + t * d
        dist = np.sum((p - proj) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest = proj
    return closest

def triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by p1, p2, p3."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    min_area = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective function for optimization: negative of min triangle area."""
    points = x.reshape(N, 2)
    # Project points to triangle if outside
    points = np.array([project_to_triangle(p) for p in points])
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    
    # Initial configurations: mix of boundary and interior points
    # Strategy: place some points on edges/corners, others inside
    rng = np.random.RandomState(42)
    
    best_points = None
    best_area = 0
    
    # Try multiple initializations
    n_trials = 0
    max_trials = 50
    
    while time.time() - start_time < TIME_LIMIT - 10 and n_trials < max_trials:
        n_trials += 1
        
        # Generate initial configuration
        if n_trials % 3 == 1:
            # Boundary-heavy initialization
            # Place points on the three edges and corners
            points = []
            # Corners
            points.append(A.copy())
            points.append(B.copy())
            points.append(C.copy())
            # Points on edges
            for t in [0.25, 0.5, 0.75]:
                points.append(A + t * (B - A))  # Bottom edge
            for t in [0.25, 0.5]:
                points.append(B + t * (C - B))  # Right edge
            for t in [0.25]:
                points.append(C + t * (A - C))  # Left edge
            # One interior point
            points.append(np.array([0.5, np.sqrt(3) / 6]))  # Centroid
            points = np.array(points[:N])
        elif n_trials % 3 == 2:
            # Grid-like initialization
            # Use a triangular grid pattern
            points = []
            # Layer 0: corners
            points.append(A.copy())
            points.append(B.copy())
            points.append(C.copy())
            # Layer 1: midpoints of edges
            points.append((A + B) / 2)
            points.append((B + C) / 2)
            points.append((C + A) / 2)
            # Layer 2: interior points
            # Centroid
            points.append((A + B + C) / 3)
            # Points at 1/3 and 2/3 along medians
            for vertex in [A, B, C]:
                opposite = [v for v in [A, B, C] if not np.allclose(v, vertex)][0]
                mid_opposite = (opposite + [v for v in [A, B, C] if not np.allclose(v, vertex) and not np.allclose(v, opposite)][0]) / 2
                points.append(vertex + 0.5 * (mid_opposite - vertex))
            # Trim to N
            points = np.array(points[:N])
        else:
            # Random initialization with some boundary points
            points = []
            # 3-4 boundary points
            for _ in range(rng.randint(3, 4)):
                # Random point on a random edge
                edge = rng.randint(0, 2)
                if edge == 0:
                    t = rng.random()
                    points.append(A + t * (B - A))
                elif edge == 1:
                    t = rng.random()
                    points.append(B + t * (C - B))
                else:
                    t = rng.random()
                    points.append(C + t * (A - C))
            # Remaining interior points
            while len(points) < N:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = A + u * (B - A) + v * (C - A)
                    points.append(p)
            points = np.array(points[:N])
        
        # Optimize from this initial configuration
        x0 = points.flatten()
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < objective(x0):
                opt_points = result.x.reshape(N, 2)
                opt_points = np.array([project_to_triangle(p) for p in opt_points])
                area = min_triangle_area(opt_points)
                
                if area > best_area:
                    best_area = area
                    best_points = opt_points.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 15:
            break
    
    # If no optimization succeeded, use the best initial configuration
    if best_points is None:
        # Fallback: use a reasonable static configuration
        points = [
            A, B, C,
            (A + B) / 2, (B + C) / 2, (C + A) / 2,
            (A + B + C) / 3,
            A + 0.5 * (B + C) / 2 - 0.5 * A,
            B + 0.5 * (C + A) / 2 - 0.5 * B,
            C + 0.5 * (A + B) / 2 - 0.5 * C,
            (A + B + C) / 3 + np.array([0.1, 0.05])
        ]
        best_points = np.array(points[:N])
        best_points = np.array([project_to_triangle(p) for p in best_points])
    
    return best_points
