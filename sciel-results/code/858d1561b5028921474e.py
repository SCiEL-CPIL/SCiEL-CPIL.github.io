"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p):
    """Check if point p is inside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    if y < -1e-10:
        return False
    if y + np.sqrt(3) * x - np.sqrt(3) > 1e-10:
        return False
    if y - np.sqrt(3) * x > 1e-10:
        return False
    return True

def _objective(x_flat):
    """Objective function: negative of the minimum triangle area, plus penalty for out-of-bounds."""
    points = x_flat.reshape(-1, 2)
    
    penalty = 0.0
    for p in points:
        if not _inside_triangle(p):
            penalty += 1e5
            
    if penalty > 0:
        return penalty
        
    min_area = _min_triangle_area(points)
    return -min_area

def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    # Barycentric coords: a=1-u-v, b=u, c=v is wrong for uniform distribution.
    # Correct uniform:
    # u1, u2 uniform [0,1]
    # if u1+u2 > 1, u1=1-u1, u2=1-u2
    # a = u1, b = u2, c = 1-u1-u2
    u1 = rng.random()
    u2 = rng.random()
    if u1 + u2 > 1:
        u1 = 1 - u1
        u2 = 1 - u2
    a = u1
    b = u2
    c = 1 - u1 - u2
    
    # Vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = a * 0 + b * 1 + c * 0.5
    y = a * 0 + b * 0 + c * (np.sqrt(3) / 2)
    return [x, y]

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the smallest triangle area."""
    
    best_min_area = -1e9
    best_points = None
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random starts
    for _ in range(20):
        p = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        initial_configs.append(p)
        
    # 2. Structured start: Triangular lattice approximation
    # For N=11, a 4x4 grid (16 pts) is too dense. 
    # Let's try a 3x3 grid (9 pts) + 2 random, or just a scaled lattice.
    # A known good configuration for 11 points is often related to a hexagonal packing.
    # Let's just rely on the random starts being diverse enough with Nelder-Mead.
    
    # Optimization parameters
    options = {'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7}
    
    for init_points in initial_configs:
        x0 = init_points.flatten()
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options=options)
            if res.fun < best_min_area:
                min_area = -res.fun
                # Verify validity
                pts = res.x.reshape(-1, 2)
                if all(_inside_triangle(p) for p in pts):
                    best_min_area = min_area
                    best_points = pts
        except Exception:
            continue
            
    if best_points is None:
        # Fallback
        return np.array([_random_point_in_triangle(rng) for _ in range(N)])
        
    return best_points
# EVOLVE-BLOCK-END
