"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

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
    for i, j, k in combinations(range(n), 3):
        area = triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def objective(x):
    """
    Objective function for minimization.
    x is a flat array of length 2*N.
    Returns negative of min triangle area + penalty for out-of-bounds.
    """
    points = x.reshape(N, 2)
    
    # Check bounds and penalty
    penalty = 0.0
    for i in range(N):
        p = points[i]
        if not point_in_triangle(p):
            penalty += 100.0
            
    if penalty > 0:
        return penalty
    
    min_area = min_triangle_area(points)
    return -min_area

def local_refine(points, steps=1000):
    """Simple local refinement using random perturbations (like a short SA or hill climbing)."""
    current = points.copy()
    current_score = min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    for _ in range(steps):
        idx = np.random.randint(0, N)
        old_point = current[idx].copy()
        # Small perturbation
        delta = np.random.normal(0, 0.01, 2)
        new_point = old_point + delta
        
        if point_in_triangle(new_point):
            current[idx] = new_point
            new_score = min_triangle_area(current)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = current.copy()
            else:
                current[idx] = old_point
                
    return best

def construct():
    np.random.seed(42)
    
    # Bounds for differential evolution
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    
    # Differential Evolution
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            tol=1e-8, 
            seed=42,
            polish=False, # Disable built-in polish to avoid issues with non-smooth objective
            init='latinhypercube'
        )
        
        best_points = result.x.reshape(N, 2)
        
        # Local refinement to polish the result
        refined_points = local_refine(best_points, steps=2000)
        
        # Final verification
        valid_points = []
        for p in refined_points:
            if point_in_triangle(p):
                valid_points.append(p)
            else:
                centroid = (V1 + V2 + V3) / 3
                valid_points.append(centroid)
                
        return np.array(valid_points)
        
    except Exception as e:
        # Fallback
        points = []
        y0 = 0.05
        for i in range(4):
            x = 0.1 + i * 0.2
            points.append(np.array([x, y0]))
        y1 = np.sqrt(3)/4
        for i in range(4):
            x = 0.2 + i * 0.2
            points.append(np.array([x, y1]))
        y2 = np.sqrt(3)/2 * 0.8
        for i in range(3):
            x = 0.35 + i * 0.15
            points.append(np.array([x, y2]))
        return np.array(points)
# EVOLVE-BLOCK-END
