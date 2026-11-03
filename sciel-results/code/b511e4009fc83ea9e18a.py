import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: array of shape (N, 2)
    """
    # points shape (11, 2)
    # We need min over all i < j < k of 0.5 * |det(Pj-Pi, Pk-Pi)|
    
    # Vectorized approach for speed
    # Expand points to compute all pair differences? 
    # For N=11, simple loops are fine, but let's try to be efficient.
    
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """
    Objective function for minimization.
    We want to maximize min_area, so we minimize -min_area.
    x: flat array of 22 coordinates.
    """
    points = x.reshape(-1, 2)
    # Ensure points stay inside triangle? 
    # Nelder-Mead might step outside. 
    # Penalty method: if any point is outside, return large penalty.
    # Check if point is inside triangle V0, V1, V2
    # Barycentric coordinates or cross products.
    # Cross product test:
    # For CCW triangle V0, V1, V2:
    # sign of cross(V1-V0, P-V0) should be >= 0
    # sign of cross(V2-V1, P-V1) should be >= 0
    # sign of cross(V0-V2, P-V2) should be >= 0
    
    # Pre-compute edge vectors
    # e1 = V1-V0 = (1, 0)
    # e2 = V2-V1 = (-0.5, sqrt(3)/2)
    # e3 = V0-V2 = (-0.5, -sqrt(3)/2)
    
    # Check all points
    for p in points:
        # Cross with e1
        if (V1[0]-V0[0])*(p[1]-V0[1]) - (V1[1]-V0[1])*(p[0]-V0[0]) < -1e-9:
            return 1e6 # Penalty
        # Cross with e2
        if (V2[0]-V1[0])*(p[1]-V1[1]) - (V2[1]-V1[1])*(p[0]-V1[0]) < -1e-9:
            return 1e6
        # Cross with e3
        if (V0[0]-V2[0])*(p[1]-V2[1]) - (V0[1]-V2[1])*(p[0]-V2[0]) < -1e-9:
            return 1e6
            
    area = min_triangle_area(points)
    return -area

def is_inside_triangle(p):
    # Same checks as above, return bool
    if (V1[0]-V0[0])*(p[1]-V0[1]) - (V1[1]-V0[1])*(p[0]-V0[0]) < -1e-9:
        return False
    if (V2[0]-V1[0])*(p[1]-V1[1]) - (V2[1]-V1[1])*(p[0]-V1[0]) < -1e-9:
        return False
    if (V0[0]-V2[0])*(p[1]-V2[1]) - (V0[1]-V2[1])*(p[0]-V2[0]) < -1e-9:
        return False
    return True

def generate_random_points(seed):
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1.0:
            p = V0 + u*(V1-V0) + v*(V2-V0)
            points.append(p)
    return np.array(points).flatten()

def generate_grid_points():
    # Generate points on a triangular lattice
    # Approximate 11 points. 
    # A 3x3 grid in triangular coords has 1+2+3+... points?
    # Let's just place points on a grid and filter.
    # Or use a specific pattern.
    # For now, use a denser random search with more restarts.
    # The "Grid" idea is good but mapping to exactly 11 points is tricky.
    # Let's use "Fibonacci Spiral" in the triangle.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations
    # 1. Random restarts
    # 2. "Grid-like" restarts: Perturb a regular hexagonal pattern
    
    # Pre-defined good patterns?
    # Let's rely on Nelder-Mead with many random restarts.
    # N=11, dimension 22.
    
    num_restarts = 50
    # We can dynamically increase restarts if time permits.
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        x0 = generate_random_points(seed=i+1)
        
        # Optional: Local improvement of x0 before NM?
        # NM is good enough.
        
        try:
            # Bounds are handled by penalty, but we can provide initial feasible point.
            # generate_random_points ensures feasibility.
            
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            
            if res.success or res.fun < best_score:
                # Check if result is valid (inside triangle)
                pts = res.x.reshape(-1, 2)
                valid = all(is_inside_triangle(p) for p in pts)
                if valid:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to seed-like behavior
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(2)
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
        
    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will place it in the block.
# EVOLVE-BLOCK-END
