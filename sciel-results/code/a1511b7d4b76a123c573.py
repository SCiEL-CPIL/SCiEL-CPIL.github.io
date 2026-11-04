"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import math

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (2D points)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        a = _triangle_area(points[i], points[j], points[k])
        if a < min_area:
            min_area = a
    return min_area

def _inside_triangle(p, margin=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # y >= 0
    # y <= sqrt(3) * x  (left edge from (0,0) to (0.5, sqrt(3)/2))
    # y <= -sqrt(3) * (x - 1)  (right edge from (1,0) to (0.5, sqrt(3)/2))
    h = math.sqrt(3) / 2
    if y < -margin:
        return False
    if x < -margin or x > 1 + margin:
        return False
    # Left edge: from (0,0) to (0.5, h), line: y = 2*sqrt(3)*x for x in [0, 0.5]
    # Actually the left edge goes from (0,0) to (0.5, h). The line equation:
    # slope = h / 0.5 = 2*h = sqrt(3)
    # So y <= sqrt(3) * x is NOT the left edge constraint.
    # Left edge: points satisfy y <= sqrt(3) * x? Let's check: at x=0.5, y should be <= h = sqrt(3)/2.
    # sqrt(3)*0.5 = sqrt(3)/2 = h. Yes. At x=0, y<=0. Yes.
    # So left constraint: y <= sqrt(3) * x
    # Right edge: from (1,0) to (0.5, h). slope = (h-0)/(0.5-1) = h/(-0.5) = -2h = -sqrt(3)
    # Line: y - 0 = -sqrt(3)*(x - 1), so y = -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    # Constraint: y <= sqrt(3)*(1-x)
    if y > math.sqrt(3) * x + margin:
        return False
    if y > math.sqrt(3) * (1 - x) + margin:
        return False
    return True

def _project_to_triangle(p, margin=1e-8):
    """Project point p onto the triangle (0,0),(1,0),(0.5,sqrt(3)/2) with margin."""
    x, y = p
    h = math.sqrt(3) / 2
    # Clip to triangle
    # y >= 0
    if y < margin:
        y = margin
    # y <= sqrt(3)*x  => x >= y/sqrt(3)
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    # y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    # Also x >= 0, x <= 1
    if x < margin:
        x = margin
    if x > 1 - margin:
        x = 1 - margin
    return [x, y]

def _neg_min_area_flat(flat_points):
    """Negative of minimum triangle area for optimization (minimize)."""
    points = flat_points.reshape(N, 2)
    # Project each point to be inside triangle
    points = np.array([_project_to_triangle(p) for p in points])
    min_a = _min_triangle_area(points)
    return -min_a

def construct():
    """Construct 11 points in the equilateral triangle maximizing min triangle area."""
    # Start from a reasonable initial configuration:
    # Use a grid-like pattern plus corners and center
    h = math.sqrt(3) / 2
    
    # Initialize with a structured pattern
    initial = []
    # Corners
    initial.append([0.05, 0.05])
    initial.append([0.95, 0.05])
    initial.append([0.5, h - 0.05])
    # Center
    initial.append([0.5, h / 3])
    # Midpoints of edges
    initial.append([0.25, h / 6])
    initial.append([0.75, h / 6])
    initial.append([0.5, 2 * h / 3])
    # Additional interior points
    initial.append([0.3, h / 3])
    initial.append([0.7, h / 3])
    initial.append([0.5, h / 4])
    initial.append([0.4, 2*h/5])
    
    # Ensure all are inside
    initial = np.array([_project_to_triangle(p) for p in initial])
    
    # Optimize
    best_flat = initial.flatten()
    best_val = _neg_min_area_flat(best_flat)
    
    # Try multiple random starts and local optimization
    rng = np.random.RandomState(42)
    for trial in range(8):
        if trial == 0:
            flat0 = initial.flatten()
        else:
            # Random start inside triangle
            pts = []
            for _ in range(N):
                u, v = rng.random(2)
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                p = [u + 0.5 * v, h * v]
                pts.append(_project_to_triangle(p))
            flat0 = np.array(pts).flatten()
        
        res = minimize(_neg_min_area_flat, flat0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_flat = res.x
    
    # Final points
    points = best_flat.reshape(N, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    
    return points
# EVOLVE-BLOCK-END
