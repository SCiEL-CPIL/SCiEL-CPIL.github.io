"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Vertices of the equilateral triangle with area 1
# (0,0), (1,0), (0.5, sqrt(3)/2)
# Area = 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4
# Wait, the task says "unit-area triangle".
# The seed program uses vertices (0,0), (1,0), (1/2, sqrt(3)/2).
# Let's check the area of this triangle.
# Area = 0.5 * 1 * (sqrt(3)/2) = sqrt(3)/4 ≈ 0.433.
# The task says "divides by its area".
# The evaluator measures the "raw value".
# "search score is value/seed".
# The seed program returns points in the triangle defined by (0,0), (1,0), (0.5, sqrt(3)/2).
# If the evaluator calculates the min area of 11 points and divides by the area of the triangle,
# then the "raw value" is the normalized minimum area.
# Let's assume the evaluator computes: min_area(points) / area(triangle).
# We want to maximize this ratio.

def _triangle_area(p1, p2, p3):
    """Calculate the area of a triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _all_triangle_areas(points):
    """Calculate areas of all triangles formed by 3 points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _min_area(points):
    """Calculate the minimum triangle area for a set of points."""
    return np.min(_all_triangle_areas(points))

def _is_inside_triangle(p, t1, t2, t3, eps=1e-9):
    """Check if point p is inside triangle t1-t2-t3."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, t1, t2)
    d2 = sign(p, t2, t3)
    d3 = sign(p, t3, t1)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def _project_to_triangle(p, t1, t2, t3):
    """Project point p to the closest point on or inside triangle t1-t2-t3."""
    if _is_inside_triangle(p, t1, t2, t3):
        return p
    
    # Check edges
    def closest_point_on_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab
    
    c1 = closest_point_on_segment(p, t1, t2)
    c2 = closest_point_on_segment(p, t2, t3)
    c3 = closest_point_on_segment(p, t3, t1)
    
    d1 = np.linalg.norm(p - c1)
    d2 = np.linalg.norm(p - c2)
    d3 = np.linalg.norm(p - c3)
    
    if d1 < d2 and d1 < d3:
        return c1
    elif d2 < d1 and d2 < d3:
        return c2
    else:
        return c3

def _enforce_boundary(points, t1, t2, t3):
    """Ensure all points are inside the triangle."""
    for i in range(len(points)):
        points[i] = _project_to_triangle(points[i], t1, t2, t3)
    return points

# The objective function for optimization
# We want to maximize the minimum area.
# So we minimize the negative of the minimum area.
def objective(x):
    # Reshape x to (11, 2)
    points = x.reshape(-1, 2)
    
    # Enforce boundary (optional, but helps with numerical stability)
    # It's better to let the optimizer handle it with bounds if possible,
    # but the triangle is not a box.
    # Using a penalty or projection.
    # Let's use projection to keep points inside.
    points = _enforce_boundary(points, t1, t2, t3)
    
    # Calculate min area
    min_a = _min_area(points)
    
    return -min_a

# Triangle vertices
t1 = np.array([0.0, 0.0])
t2 = np.array([1.0, 0.0])
t3 = np.array([0.5, np.sqrt(3) / 2])

# Area of the triangle
area_tri = _triangle_area(t1, t2, t3)

# Bounds for differential evolution
# x in [0, 1], y in [0, sqrt(3)/2]
bounds = [(0.0, 1.0)] * 11 + [(0.0, np.sqrt(3) / 2)] * 11

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial guess: Hexagonal grid or similar structured layout
    # A good starting point is a grid or random.
    # Let's try a few initial points.
    
    # 1. Random initialization
    np.random.seed(42)
    x0 = np.random.uniform(0.05, 0.95, (11, 2))
    x0[:, 1] *= np.sqrt(3) / 2
    
    # Ensure inside
    x0 = _enforce_boundary(x0, t1, t2, t3).flatten()
    
    best_x = x0
    best_val = objective(x0)
    
    # Differential Evolution
    # We have 22 variables.
    # DE is robust.
    try:
        # maxiter is limited by time
        # Let's run DE for a while
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=True,  # Use Nelder-Mead to polish
            init='sobol'
        )
        
        if time.time() < deadline:
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        points = best_x.reshape(-1, 2)
        points = _enforce_boundary(points, t1, t2, t3)
        return points
    
    # Refine with Nelder-Mead if time permits
    # The 'polish' in DE already does this, but we can do a dedicated pass.
    
    # Let's try a few more local optimizations from different starts if time permits
    # Or just polish the best one found.
    
    # Actually, DE with polish=True is quite good.
    # Let's do one more high-precision polish on the best solution found so far.
    
    try:
        result2 = minimize(
            objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000}
        )
        
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    except Exception:
        pass
    
    points = best_x.reshape(-1, 2)
    points = _enforce_boundary(points, t1, t2, t3)
    
    return points

# EVOLVE-BLOCK-END
