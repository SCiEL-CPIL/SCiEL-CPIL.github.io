import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11

def point_in_triangle(p, a, b, c):
    # Barycentric coordinate technique
    # p is inside if all barycentric coords are in [0, 1]
    # Using cross products for 2D
    v0 = np.array(c) - np.array(a)
    v1 = np.array(b) - np.array(a)
    v2 = np.array(p) - np.array(a)
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return False
        
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def min_triangle_area(points):
    # points: (N, 2)
    n = len(points)
    min_area = float('inf')
    # Precompute cross products? No, just iterate. 165 iterations is trivial.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    points = x.reshape(-1, 2)
    # Check bounds
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    # Penalty if out of bounds
    for p in points:
        if not point_in_triangle(p, a, b, c):
            return 1e6 # Large penalty, we want to minimize this (which is -area + penalty)
            
    area = min_triangle_area(points)
    return -area

def construct():
    # Vertices of the equilateral triangle
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    best_points = None
    best_score = -1.0
    
    # Run multiple restarts to avoid local minima
    num_restarts = 5
    for i in range(num_restarts):
        # Initial guess: Hexagonal-ish grid or Random
        if i == 0:
            # Try a structured grid
            # 3 rows: 4, 4, 3
            pts = []
            # Row 1 (y=0.1): 4 points
            for j in range(4):
                x = 0.1 + j * 0.25
                pts.append([x, 0.1])
            # Row 2 (y=0.3): 4 points
            for j in range(4):
                x = 0.15 + j * 0.25
                pts.append([x, 0.3])
            # Row 3 (y=0.5): 3 points
            for j in range(3):
                x = 0.2 + j * 0.3
                pts.append([x, 0.5])
            x0 = np.array(pts).flatten()
        else:
            # Random initialization, but ensure they are inside
            pts = []
            for _ in range(N):
                while True:
                    u = np.random.random()
                    v = np.random.random()
                    if u + v < 1.0:
                        p = u * b + v * c
                        pts.append(p)
                        break
            x0 = np.array(pts).flatten()
        
        # Optimize
        # Nelder-Mead is good for low-dimensional, non-smooth problems
        # maxiter set high enough to converge, but bounded by time
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < 1e5: # Check if it didn't fail due to penalty
            current_area = -res.fun
            if current_area > best_score:
                best_score = current_area
                best_points = res.x.reshape(-1, 2)
                
    # Fallback if optimization failed or returned bad result
    if best_points is None:
        # Return the seed-like random points
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
        
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 11

def point_in_triangle(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return False
        
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    points = x.reshape(-1, 2)
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    for p in points:
        if not point_in_triangle(p, a, b, c):
            return 1e6
            
    area = min_triangle_area(points)
    return -area

def construct():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    best_points = None
    best_score = -1.0
    
    num_restarts = 4
    for i in range(num_restarts):
        if i == 0:
            # Structured initial guess
            pts = []
            # Approximate hexagonal packing
            # Row 1: 4 points
            for j in range(4):
                pts.append([0.1 + j * 0.25, 0.1])
            # Row 2: 4 points
            for j in range(4):
                pts.append([0.15 + j * 0.25, 0.35])
            # Row 3: 3 points
            for j in range(3):
                pts.append([0.2 + j * 0.3, 0.6])
            x0 = np.array(pts).flatten()
        else:
            # Random restarts
            rng = np.random.RandomState(42 + i)
            pts = []
            for _ in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v < 1.0:
                        p = u * b + v * c
                        pts.append(p)
                        break
            x0 = np.array(pts).flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
            
            if res.fun < 1e5:
                current_area = -res.fun
                if current_area > best_score:
                    best_score = current_area
                    best_points = res.x.reshape(-1, 2)
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to simple random
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
        
    return best_points
# EVOLVE-BLOCK-END
