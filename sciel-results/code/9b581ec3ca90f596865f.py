"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])
CENTROID = (V1 + V2 + V3) / 3

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    # Pre-compute indices for speed
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective function for minimization: negative of min triangle area."""
    points = x.reshape((N, 2))
    # Check if all points are inside the triangle
    # If not, return a large penalty
    for p in points:
        if not point_in_triangle(p):
            return 1e6
    return -min_triangle_area(points)

def initial_configuration():
    """Create a structured initial configuration for 11 points in the triangle."""
    points = []
    # Bottom row: 4 points
    y0 = 0.1
    for i in range(4):
        x = 0.15 + i * 0.2
        points.append([x, y0])
    # Middle row: 4 points
    y1 = 0.4
    for i in range(4):
        x = 0.25 + i * 0.2
        points.append([x, y1])
    # Top row: 3 points
    y2 = 0.7
    for i in range(3):
        x = 0.35 + i * 0.2
        points.append([x, y2])
    
    # Ensure they are inside
    valid_points = []
    for p in points:
        p_arr = np.array(p)
        if point_in_triangle(p_arr):
            valid_points.append(p_arr)
        else:
            # Move towards center
            p_new = CENTROID + 0.8 * (p_arr - CENTROID)
            valid_points.append(p_new)
            
    return np.array(valid_points).flatten()

def construct():
    best_score = -1e9
    best_points = None
    
    # Define initial configurations
    inits = []
    base_init = initial_configuration()
    inits.append(base_init)
    
    # Add some noisy inits
    np.random.seed(123)
    for _ in range(3):
        noise = np.random.normal(0, 0.05, (N, 2))
        p_noisy = base_init.reshape((N, 2)) + noise
        # Ensure validity
        valid_p = []
        for pt in p_noisy:
            if point_in_triangle(pt):
                valid_p.append(pt)
            else:
                pt_new = CENTROID + 0.8 * (pt - CENTROID)
                valid_p.append(pt_new)
        inits.append(np.array(valid_p).flatten())

    # Run Nelder-Mead on each
    for x0 in inits:
        options = {
            'maxiter': 2000,
            'xatol': 1e-5,
            'fatol': 1e-5
        }
        try:
            result = minimize(objective, x0, method='Nelder-Mead', options=options)
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = result.x.reshape((N, 2))
        except Exception:
            pass
            
    if best_points is None:
        return base_init.reshape((N, 2))
        
    return best_points
# EVOLVE-BLOCK-END
