"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # i, j, k indices
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    points = x.reshape(-1, 2)
    # Penalize points outside triangle
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 1.0
    return -min_triangle_area(points) + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    def initial_points(seed_offset=0):
        # Grid-based initialization in barycentric coordinates
        # Divide the triangle into smaller triangles
        # We want 11 points. A 3x3 grid has 6 points, 4x4 has 10, 5x5 has 15.
        # Let's try a structured set of points.
        # Barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0.
        # p = u*V0 + v*V1 + w*V2
        
        pts = []
        
        # Try a few different grid densities and jitter
        # For 11 points, we can pick specific spots.
        # Or generate a grid of size 4 (10 points) and add 1 more?
        # Or just use random points but constrained to be well-distributed?
        
        # Strategy: Generate points on a grid in the (u,v) plane where u>=0, v>=0, u+v<=1.
        # Map to Cartesian.
        
        # Let's try 3 different initialization strategies per restart
        strategy = seed_offset % 3
        
        if strategy == 0:
            # 3x3 grid + 2 extra
            grid_size = 3
            for i in range(grid_size):
                for j in range(grid_size - i):
                    u = (i + 0.5) / (grid_size + 1)
                    v = (j + 0.5) / (grid_size + 1)
                    w = 1 - u - v
                    if w > 0:
                        p = u * V0 + v * V1 + w * V2
                        # Add jitter
                        p += rng.uniform(-0.05, 0.05, 2)
                        pts.append(p)
            # Add 2 more points randomly
            for _ in range(2):
                u = rng.random()
                v = rng.random() * (1 - u)
                w = 1 - u - v
                p = u * V0 + v * V1 + w * V2
                pts.append(p)
        elif strategy == 1:
            # 4x4 grid (10 points) + 1 centroid
            grid_size = 4
            for i in range(grid_size):
                for j in range(grid_size - i):
                    u = (i + 0.5) / (grid_size + 1)
                    v = (j + 0.5) / (grid_size + 1)
                    w = 1 - u - v
                    if w > 0:
                        p = u * V0 + v * V1 + w * V2
                        p += rng.uniform(-0.02, 0.02, 2)
                        pts.append(p)
            pts.append(V0 + V1 + V2) # Centroid
        else:
            # Random but with minimum distance constraint (simple rejection)
            pts = []
            while len(pts) < 11:
                u = rng.random()
                v = rng.random() * (1 - u)
                w = 1 - u - v
                p = u * V0 + v * V1 + w * V2
                # Check distance to existing points
                if all(np.linalg.norm(p - q) > 0.1 for q in pts):
                    pts.append(p)
                else:
                    # If stuck, just add random
                    if len(pts) > 0 and rng.random() < 0.5:
                         pts.append(p)
        
        # Ensure we have exactly 11 points
        while len(pts) < 11:
            u = rng.random()
            v = rng.random() * (1 - u)
            w = 1 - u - v
            p = u * V0 + v * V1 + w * V2
            pts.append(p)
        
        return np.array(pts[:11])
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple initial configurations
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        points = initial_points(restart)
        x0 = points.flatten()
        
        # Nelder-Mead optimization
        try:
            result = minimize(
                negative_min_area_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        optimized_points = result.x.reshape(-1, 2)
        
        # Check if all points are inside triangle
        valid = all(point_in_triangle(p) for p in optimized_points)
        
        if valid:
            score = min_triangle_area(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points
    
    if best_points is None:
        # Fallback: return initial points
        best_points = initial_points(0)
    
    return best_points
# EVOLVE-BLOCK-END
