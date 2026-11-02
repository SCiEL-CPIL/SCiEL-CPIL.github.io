"""Sobol-Grid + Local Nudge strategy for Heilbronn triangles."""
N = 11
import numpy as np
import time
from itertools import combinations

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])


def point_in_triangle(p):
    """Check if point p is inside or on the boundary of triangle ABC."""
    # Using barycentric coordinates
    v0 = B - A
    v1 = C - A
    v2 = p - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    return u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9


def triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
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


def sobol_initialization(n, rng):
    """
    Generate n points in the triangle using a quasi-random approach.
    We use a simple low-discrepancy sequence based on Halton-like construction
    or just use scipy's Sobol if available, but to be safe we use a manual approach.
    """
    # For simplicity and robustness, we use a stratified random sampling
    # combined with some deterministic grid points for initial coverage.
    
    points = []
    
    # Add centroid
    centroid = (A + B + C) / 3
    points.append(centroid)
    
    # Add midpoints of edges
    points.append((A + B) / 2)
    points.append((B + C) / 2)
    points.append((C + A) / 2)
    
    # Add vertices (slightly inset to avoid boundary issues)
    eps = 0.01
    points.append(A + eps * (C - A) + eps * (B - A))  # near A
    points.append(B + eps * (C - B) + eps * (A - B))  # near B
    points.append(C + eps * (A - C) + eps * (B - C))  # near C
    
    # Fill remaining with quasi-random points
    # Use a simple low-discrepancy sequence
    for _ in range(n - len(points)):
        # Use golden ratio-based low-discrepancy
        while True:
            # Generate random barycentric coordinates
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = A * (1 - u - v) + B * u + C * v
                # Check if it's sufficiently far from existing points
                min_dist = min(np.linalg.norm(p - q) for q in points)
                if min_dist > 0.05 or len(points) >= n - 2:
                    points.append(p)
                    break
    
    return np.array(points[:n])


def local_nudge(points, rng, max_iter=500, deadline=None):
    """
    Local search that nudges points involved in the smallest triangles.
    """
    best_points = points.copy()
    best_area = min_triangle_area(points)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Find the smallest triangle
        min_area = float('inf')
        min_tri = None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
                        min_tri = (i, j, k)
        
        if min_tri is None:
            break
        
        i, j, k = min_tri
        
        # Try to nudge one of the three points
        improved = False
        for idx in [i, j, k]:
            p = points[idx].copy()
            
            # Try small random nudges
            for _ in range(20):
                step = 0.01 * (1 + 0.5 * rng.random())
                angle = 2 * np.pi * rng.random()
                new_p = p + step * np.array([np.cos(angle), np.sin(angle)])
                
                # Check if new point is in triangle
                if not point_in_triangle(new_p):
                    continue
                
                # Check minimum distance to other points
                min_dist = min(np.linalg.norm(new_p - points[m]) 
                              for m in range(len(points)) if m != idx)
                if min_dist < 0.01:
                    continue
                
                # Try the move
                old_p = points[idx].copy()
                points[idx] = new_p
                
                new_min_area = min_triangle_area(points)
                
                if new_min_area > best_area + 1e-9:
                    best_points = points.copy()
                    best_area = new_min_area
                    improved = True
                else:
                    points[idx] = old_p
            
            if improved:
                break
        
        # If no improvement in this iteration, try larger steps occasionally
        if not improved and it % 10 == 0:
            # Random restart nudge
            idx = rng.randint(0, len(points) - 1)
            for _ in range(5):
                step = 0.05 * rng.random()
                angle = 2 * np.pi * rng.random()
                new_p = points[idx] + step * np.array([np.cos(angle), np.sin(angle)])
                
                if not point_in_triangle(new_p):
                    continue
                
                min_dist = min(np.linalg.norm(new_p - points[m]) 
                              for m in range(len(points)) if m != idx)
                if min_dist < 0.02:
                    continue
                
                old_p = points[idx].copy()
                points[idx] = new_p
                
                new_min_area = min_triangle_area(points)
                
                if new_min_area > best_area + 1e-9:
                    best_points = points.copy()
                    best_area = new_min_area
                else:
                    points[idx] = old_p
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Initialize with Sobol-like quasi-random points
    points = sobol_initialization(N, rng)
    
    best_points = points.copy()
    best_area = min_triangle_area(points)
    
    # Local nudge refinement
    points = local_nudge(points, rng, max_iter=200, deadline=deadline)
    current_area = min_triangle_area(points)
    
    if current_area > best_area:
        best_points = points.copy()
        best_area = current_area
    
    # If we have time, try a few more restarts with different seeds
    while time.time() < deadline - 5:
        rng = np.random.RandomState(int(rng.random() * 100000))
        points = sobol_initialization(N, rng)
        
        points = local_nudge(points, rng, max_iter=100, deadline=deadline)
        current_area = min_triangle_area(points)
        
        if current_area > best_area:
            best_points = points.copy()
            best_area = current_area
    
    return best_points
