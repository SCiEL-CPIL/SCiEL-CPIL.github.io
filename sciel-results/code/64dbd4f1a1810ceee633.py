"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Vertices of the equilateral triangle
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def is_inside(points):
    # points: (N, 2)
    x = points[:, 0]
    y = points[:, 1]
    # y >= 0
    if np.any(y < -1e-9):
        return False
    # y <= sqrt(3) * x
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    # sqrt(3) * x + y <= sqrt(3)
    if np.any(np.sqrt(3) * x + y > np.sqrt(3) + 1e-9):
        return False
    return True

def min_area(points):
    # points: (N, 2)
    min_a = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_a:
                    min_a = area
    return min_a

def objective(x):
    points = x.reshape(N, 2)
    if not is_inside(points):
        return 1.0 
    area = min_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_points = None
    best_score = -1.0
    rng = np.random.default_rng(7)
    
    def generate_random_points():
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1.0:
                p = (1-u-v)*V1 + u*V2 + v*V3
                pts.append(p)
        return np.array(pts)

    # Initial structured candidate: Triangular grid
    # Side 4 grid has 10 points.
    # Let's try to construct a grid-like start.
    # This is optional, random starts are often enough.
    
    # Run multiple optimizations
    num_initial_starts = 20
    
    for i in range(num_initial_starts):
        if time.time() > deadline:
            break
            
        x0 = generate_random_points().flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < 0.5: # Valid region
                final_pts = res.x.reshape(N, 2)
                if is_inside(final_pts):
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_pts
        except Exception:
            pass

    # Continue with more random restarts if time permits
    while time.time() < deadline - 5.0:
        x0 = generate_random_points().flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            
            if res.fun < 0.5:
                final_pts = res.x.reshape(N, 2)
                if is_inside(final_pts):
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        best_points = generate_random_points()

    return best_points
# EVOLVE-BLOCK-END
