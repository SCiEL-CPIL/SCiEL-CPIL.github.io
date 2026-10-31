"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def triangle_area(p1, p2, p3):
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def is_inside_triangle(p):
    x, y = p
    # Check y >= 0
    if y < 0: return False
    # Check y <= sqrt(3)*x
    if y > (3**0.5)*x: return False
    # Check y <= sqrt(3)*(1-x)
    if y > (3**0.5)*(1-x): return False
    return True

def objective(flat_points):
    points = flat_points.reshape(N, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        if not is_inside_triangle(p):
            # Calculate distance to nearest edge or just a fixed penalty
            # A simple penalty: sum of violations
            x, y = p
            if y < 0: penalty += abs(y)
            if y > (3**0.5)*x: penalty += abs(y - (3**0.5)*x)
            if y > (3**0.5)*(1-x): penalty += abs(y - (3**0.5)*(1-x))
            
    if penalty > 0:
        # Heavily penalize invalid points
        return 1000 + penalty

    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    # We want to maximize min_area, so minimize -min_area
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_score = -1
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    num_restarts = 20
    points_per_restart = 1000
    
    for r in range(num_restarts):
        # Check time
        if time.time() > deadline:
            break
            
        # Initialize points randomly inside the triangle
        # Using barycentric coordinates for uniform sampling
        u = rng.random(N)
        v = rng.random(N)
        # Rejection sampling or transformation
        # If u+v > 1, reflect: u = 1-u, v = 1-v
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        
        # A = (0,0), B = (1,0), C = (0.5, sqrt(3)/2)
        # P = u*A + v*B + (1-u-v)*C
        # x = u*0 + v*1 + (1-u-v)*0.5 = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
        # y = u*0 + v*0 + (1-u-v)*sqrt(3)/2 = (1-u-v)*sqrt(3)/2
        
        x = 0.5 + 0.5*v - 0.5*u
        y = (1 - u - v) * (3**0.5) / 2
        initial_points = np.column_stack((x, y)).flatten()
        
        # Optimize
        try:
            result = minimize(objective, initial_points, method='Nelder-Mead', 
                              options={'maxiter': points_per_restart, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Check time again
            if time.time() > deadline:
                break
                
            if result.fun < best_score:
                best_score = result.fun
                best_points = result.x.reshape(N, 2)
                
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to random if optimization failed
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = 0.5 + 0.5*v - 0.5*u
        y = (1 - u - v) * (3**0.5) / 2
        best_points = np.column_stack((x, y))
        
    return best_points
# EVOLVE-BLOCK-END
