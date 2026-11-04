"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

# Area of the container triangle
CONTAINER_AREA = np.sqrt(3) / 4.0

# Indices for all 165 triples
triples = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            triples.append((i, j, k))
triples = np.array(triples)

def get_min_area(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    # points[triples[:, 0]] gives shape (165, 2) for first point of each triple
    p1 = points[triples[:, 0]]
    p2 = points[triples[:, 1]]
    p3 = points[triples[:, 2]]
    
    # Vectorized area calculation
    # 2 * Area = | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def penalty(points):
    """
    Calculate penalty for points outside the triangle.
    Returns 0 if all points are inside.
    """
    # Convert to barycentric coordinates relative to V1, V2, V3
    # P = u*V1 + v*V2 + w*V3, u+v+w=1
    # Inverse of matrix [V1, V2, V3]
    # Let's use the standard barycentric formula
    # Area of triangle (P, V2, V3) / Total Area = u
    # Area of triangle (V1, P, V3) / Total Area = v
    # Area of triangle (V1, V2, P) / Total Area = w
    
    # Vectorized barycentric coords
    # p: (N, 2)
    # V1, V2, V3 are (2,)
    
    # u = area(P, V2, V3) / Area(V1, V2, V3)
    # 2*Area(P, V2, V3) = |(V2-P)x(V3-P)|
    
    v2_minus_p = V2[None, :] - points
    v3_minus_p = V3[None, :] - points
    cross_u = v2_minus_p[:, 0] * v3_minus_p[:, 1] - v2_minus_p[:, 1] * v3_minus_p[:, 0]
    u = cross_u / (2 * CONTAINER_AREA)
    
    v1_minus_p = V1[None, :] - points
    v3_minus_p = V3[None, :] - points
    cross_v = v1_minus_p[:, 0] * v3_minus_p[:, 1] - v1_minus_p[:, 1] * v3_minus_p[:, 0]
    v = cross_v / (2 * CONTAINER_AREA)
    
    w = 1.0 - u - v
    
    # Check if u, v, w are >= 0
    # If any are negative, point is outside.
    # Penalty is sum of squares of negative parts
    neg_u = np.clip(-u, 0, None)
    neg_v = np.clip(-v, 0, None)
    neg_w = np.clip(-w, 0, None)
    
    return np.sum(neg_u**2 + neg_v**2 + neg_w**2)

def objective(params):
    """
    params: (2*N,) array
    Returns negative min area plus penalty.
    """
    points = params.reshape(N, 2)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 1e100
        
    pen = penalty(points)
    if pen > 1e-6:
        # Strong penalty to keep points inside
        return 1e6 * pen - get_min_area(points)
        
    return -get_min_area(points)

def project_to_triangle(points):
    """
    Project points to the nearest point inside the triangle.
    Used for initialization cleanup.
    """
    # This is a simple projection. For better performance, we might just rely on the penalty.
    # But for initial guesses, it's good to start inside.
    # We can use the barycentric coords to clip and convert back.
    v2_minus_p = V2[None, :] - points
    v3_minus_p = V3[None, :] - points
    cross_u = v2_minus_p[:, 0] * v3_minus_p[:, 1] - v2_minus_p[:, 1] * v3_minus_p[:, 0]
    u = cross_u / (2 * CONTAINER_AREA)
    
    v1_minus_p = V1[None, :] - points
    v3_minus_p = V3[None, :] - points
    cross_v = v1_minus_p[:, 0] * v3_minus_p[:, 1] - v1_minus_p[:, 1] * v3_minus_p[:, 0]
    v = cross_v / (2 * CONTAINER_AREA)
    
    w = 1.0 - u - v
    
    # Clip negative values to 0 and renormalize
    u = np.clip(u, 0, None)
    v = np.clip(v, 0, None)
    w = np.clip(w, 0, None)
    
    total = u + v + w
    u /= total
    v /= total
    w /= total
    
    return u[:, None] * V1[None, :] + v[:, None] * V2[None, :] + w[:, None] * V3[None, :]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s budget
    
    best_score = 0.0
    best_points = None
    
    # Generate initial guesses
    guesses = []
    
    # 1. Random
    for _ in range(5):
        # Generate random points in square [0,1]x[0,1] and filter/project
        pts = np.random.rand(N, 2)
        # Project to triangle
        pts = project_to_triangle(pts)
        guesses.append(pts.flatten())
        
    # 2. Hexagonal Grid
    # Generate a hex grid and pick points inside
    hex_pts = []
    # Approximate grid
    for i in range(10):
        for j in range(10):
            x = i * 0.15 + (j % 2) * 0.075
            y = j * 0.13
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Filter those inside
    # Use barycentric check
    # ... (simpler: just project all and use first 11 distinct ones?)
    # Let's just project a random selection of the grid
    indices = np.random.choice(len(hex_pts), N, replace=False)
    grid_pts = hex_pts[indices]
    grid_pts = project_to_triangle(grid_pts)
    guesses.append(grid_pts.flatten())
    
    # 3. Concentric Rings
    for r in [0.2, 0.4, 0.6]:
        ring_pts = []
        # Center
        ring_pts.append([0.5, np.sqrt(3)/6]) # Centroid
        # Ring
        num_pts = 6
        angles = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
        for a in angles:
            px = 0.5 + r * np.cos(a)
            py = np.sqrt(3)/6 + r * np.sin(a)
            ring_pts.append([px, py])
        # We need 11 points. Center + 6 + 4 more?
        # Let's just take 11 random points from a dense set of concentric points
        # Or simpler: just add this as a rough guess
        pass # Skipping complex ring construction for now, random+grid is usually enough for NM to polish

    # Run optimization on each guess
    for guess in guesses:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success:
                pts = res.x.reshape(N, 2)
                # Ensure points are inside (project if slightly out due to numerical error)
                pts = project_to_triangle(pts)
                score = get_min_area(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts
                    
        except Exception as e:
            pass

    # If we have time, do a few more random restarts with shorter maxiter
    while time.time() < deadline - 5.0:
        pts = np.random.rand(N, 2)
        pts = project_to_triangle(pts)
        guess = pts.flatten()
        
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                pts = res.x.reshape(N, 2)
                pts = project_to_triangle(pts)
                score = get_min_area(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    if best_points is None:
        # Fallback to random
        best_points = np.random.rand(N, 2)
        best_points = project_to_triangle(best_points)
        
    return best_points

# EVOLVE-BLOCK-START
# (The code above is the implementation. The seed program had the block markers. 
# I will place the entire logic inside the block or replace the block content.)
# The prompt says "Keep the function names and signatures... Output the COMPLETE program".
# I will structure it such that the construct function is the main entry.
# EVOLVE-BLOCK-END
