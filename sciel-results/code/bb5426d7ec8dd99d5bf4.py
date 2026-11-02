"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def bary_to_cart(v1, v2):
        return v1 * B + v2 * C

    def min_triangle_area(points):
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(params):
        for i in range(N):
            v1, v2 = params[i]
            if v1 < 0 or v2 < 0 or v1 + v2 > 1.0:
                return 1e10
        points = [bary_to_cart(v1, v2) for v1, v2 in params]
        return -min_triangle_area(points)

    # Initial guess: Triangular grid of 10 points + Centroid
    grid_points = []
    for i in range(4):
        for j in range(4 - i):
            grid_points.append([i/3.0, j/3.0])
    grid_points.append([1/3.0, 1/3.0])
    
    initial_params = np.array(grid_points)
    
    best_val = float('inf')
    best_params = initial_params
    
    rng = np.random.default_rng(42)
    
    # Try the grid start
    res = minimize(objective, initial_params, method='Nelder-Mead', 
                   options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 3000, 'maxfev': 3000})
    if res.fun < best_val:
        best_val = res.fun
        best_params = res.x
        
    # Try random restarts
    for _ in range(100):
        random_params = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(2)
            if u + v > 1:
                v = 1 - u
            random_params[i] = [u, v]
        
        res = minimize(objective, random_params, method='Nelder-Mead', 
                       options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 2000, 'maxfev': 2000})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    final_points = np.array([bary_to_cart(v1, v2) for v1, v2 in best_params])
    
    return final_points
# EVOLVE-BLOCK-END
