"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools
import random

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle


def point_in_triangle(p, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective(x):
    """Negative of minimum triangle area (we minimize negative = maximize min area)."""
    points = x.reshape(-1, 2)
    # Check if all points are in triangle
    for p in points:
        if not point_in_triangle(p, eps=1e-6):
            return 1e10  # Large penalty
    return -min_triangle_area(points)


def construct():
    start_time = time.time()
    deadline = 130  # 130s out of 150s limit, safety margin
    
    best_points = None
    best_score = -1e10
    
    # Start with a reasonable initial configuration
    # Place points at vertices and centroid, then distribute others
    rng = random.Random(42)
    
    # Initial guess: symmetric arrangement
    # 3 vertices, 1 centroid, 7 more points
    init_points = [
        [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2],  # vertices
        [0.5, np.sqrt(3)/6],  # centroid
    ]
    # Add points along edges and interior
    for i in range(7):
        # Random points in triangle
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                init_points.append([u + 0.5*v, (3**0.5)/2 * v])
                break
    
    x0 = np.array(init_points).flatten()
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0_current = x0.copy()
        else:
            # Perturb the initial configuration
            x0_current = x0 + np.random.randn(2*N) * 0.1
            # Project back into triangle approximately by clamping
            for i in range(N):
                p = x0_current[2*i:2*i+2]
                if not point_in_triangle(p, eps=1e-3):
                    # Move toward centroid
                    x0_current[2*i] = 0.5 * p[0] + 0.5 * 0.5
                    x0_current[2*i+1] = 0.5 * p[1] + 0.5 * np.sqrt(3)/6
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, np.sqrt(3)/2))
        
        try:
            result = minimize(
                objective,
                x0_current,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e9:
                points = result.x.reshape(-1, 2)
                # Verify all points are in triangle
                valid = True
                for p in points:
                    if not point_in_triangle(p, eps=1e-6):
                        valid = False
                        break
                
                if valid:
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
        except Exception:
            pass
    
    # If no valid result from optimization, use initial random configuration
    if best_points is None:
        best_points = np.array(init_points)
    
    return best_points
# EVOLVE-BLOCK-END
