import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, eps=1e-9):
    # Barycentric coordinates check
    # p = u*V0 + v*V1 + w*V2
    # u = 1 - v - w
    # v = (p.x * V2.y - p.y * V2.x) / (V1.x * V2.y - V1.y * V2.x)
    # w = (p.y * V1.x - p.x * V1.y) / (V1.x * V2.y - V1.y * V2.x)
    # Denominator is 2 * Area = sqrt(3)/2
    denom = np.sqrt(3) / 2
    v = (p[0] * V2[1] - p[1] * V2[0]) / denom
    w = (p[1] * V1[0] - p[0] * V1[1]) / denom
    u = 1 - v - w
    return (u >= -eps) and (v >= -eps) and (w >= -eps)

def calculate_min_area(points):
    n = len(points)
    min_area = float('inf')
    # Use itertools.combinations for clarity, though a loop is fine for N=11
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def objective_and_penalty(x, points_shape):
    """
    x is a flattened array of 2*N coordinates.
    Returns negative min area + penalty for being outside triangle.
    """
    points = x.reshape(points_shape)
    
    # Calculate min area
    min_area = calculate_min_area(points)
    
    # Calculate penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            # Calculate distance to boundary or just a simple penalty
            # Simple penalty: sum of squares of barycentric coords that are negative
            denom = np.sqrt(3) / 2
            v = (p[0] * V2[1] - p[1] * V2[0]) / denom
            w = (p[1] * V1[0] - p[0] * V1[1]) / denom
            u = 1 - v - w
            if u < 0: penalty += u * u
            if v < 0: penalty += v * v
            if w < 0: penalty += w * w
            
    # We want to maximize min_area, so minimize -min_area
    # Add large penalty to push points inside
    return -min_area + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe within 150s
    
    best_points = None
    best_score = -float('inf')
    
    # Initial diverse starts
    # 1. Random points
    # 2. Grid-like points
    # 3. Symmetric points
    
    starts = []
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Generate random points in the triangle using rejection sampling
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                p = u * V1 + v * V2
                pts.append(p)
        starts.append(np.array(pts))
        
    # A structured start: vertices, midpoints, center
    # Vertices
    pts_struct = [V0, V1, V2]
    # Midpoints of edges
    M01 = (V0 + V1) / 2
    M12 = (V1 + V2) / 2
    M20 = (V2 + V0) / 2
    pts_struct.extend([M01, M12, M20])
    # Center
    C = (V0 + V1 + V2) / 3
    pts_struct.append(C)
    # Need 4 more. Place them symmetrically?
    # Just add random ones near the center or edges for variety
    for _ in range(4):
        u, v = rng.random(), rng.random()
        if u + v < 1:
            p = u * V1 + v * V2
            pts_struct.append(p)
    starts.append(np.array(pts_struct))
    
    points_shape = (N, 2)
    
    for i, initial_pts in enumerate(starts):
        if time.time() > deadline:
            break
            
        x0 = initial_pts.flatten()
        
        # Nelder-Mead options
        options = {
            'maxiter': 500,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'adaptive': True
        }
        
        try:
            res = minimize(objective_and_penalty, x0, args=(points_shape,), method='Nelder-Mead', options=options)
            x_opt = res.x
            points_opt = x_opt.reshape(points_shape)
            
            # Check if points are valid (inside triangle)
            valid = True
            for p in points_opt:
                if not point_in_triangle(p, eps=1e-5):
                    valid = False
                    break
            
            if valid:
                score = calculate_min_area(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt
            else:
                # If invalid, try to project or just ignore for now, 
                # but the penalty should have pushed them inside.
                # If still invalid, we can clamp them or skip.
                # Let's try to fix by projecting to nearest point in triangle?
                # For simplicity, if penalty wasn't strong enough, we might skip.
                # But 10.0 penalty factor is usually strong enough for Nelder-Mead.
                pass
                
        except Exception as e:
            pass
            
    # If no valid point was found (unlikely), return the random seed
    if best_points is None:
        # Fallback to the random generation from the original seed
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
        
    return best_points
