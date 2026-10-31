"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    if p[1] < -tol:
        return False
    if np.sqrt(3) * p[0] + p[1] > np.sqrt(3) + tol:
        return False
    if -np.sqrt(3) * p[0] + p[1] > tol:
        return False
    return True

def _is_valid(points):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p) for p in points)

def _project_to_triangle(p):
    """Project a point onto the equilateral triangle if outside."""
    x, y = float(p[0]), float(p[1])
    
    # Constraint 1: y >= 0
    if y < 0:
        y = 0.0
    
    # Constraint 2: sqrt(3)*x + y <= sqrt(3)
    val2 = np.sqrt(3) * x + y - np.sqrt(3)
    if val2 > 0:
        x -= val2 * np.sqrt(3) / 2.0
        y -= val2 / 2.0
    
    # Constraint 3: -sqrt(3)*x + y <= 0
    val3 = -np.sqrt(3) * x + y
    if val3 > 0:
        x += val3 * np.sqrt(3) / 2.0
        y -= val3 / 2.0
    
    # Clamp to valid range
    if x < 0:
        x = 0.0
    if x > 1:
        x = 1.0
    if y < 0:
        y = 0.0
    if y > np.sqrt(3) / 2:
        y = np.sqrt(3) / 2
    
    return np.array([x, y])

def _objective(points_flat):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = points_flat.reshape(-1, 2)
    
    # Check validity
    penalty = 0.0
    for p in points:
        if p[1] < 0:
            penalty += p[1] ** 2
        v2 = np.sqrt(3) * p[0] + p[1] - np.sqrt(3)
        if v2 > 0:
            penalty += (v2 / 2.0) ** 2
        v3 = -np.sqrt(3) * p[0] + p[1]
        if v3 > 0:
            penalty += (v3 / 2.0) ** 2
    
    if penalty > 1e-4:
        return 1e5 + penalty
    
    min_area = _min_triangle_area(points)
    return -min_area

def _hex_initial():
    """Generate initial points from a hexagonal/triangular lattice pattern."""
    h = np.sqrt(3) / 2
    pts = []
    
    # 4 rows: bottom to top
    # Row 0 (y=0): 4 points at x = 0, 1/3, 2/3, 1
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    
    # Row 1 (y=h/3): 3 points at x = 1/6, 1/2, 5/6
    for i in range(3):
        pts.append([1.0/6.0 + i / 3.0, h / 3.0])
    
    # Row 2 (y=2h/3): 2 points at x = 1/3, 2/3
    for i in range(2):
        pts.append([1.0/3.0 + i / 3.0, 2.0 * h / 3.0])
    
    # Row 3 (y=h): 1 point at x = 0.5
    pts.append([0.5, h])
    
    # Total: 4+3+2+1 = 10 points. Need one more.
    # Add a point in the center of the largest gap
    pts.append([0.5, h / 2.0])
    
    return np.array(pts)

def _random_initial(seed):
    """Generate random initial points inside the triangle."""
    rng = np.random.RandomState(seed)
    h = np.sqrt(3) / 2
    pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = h * v
                pts.append([x, y])
                break
    return np.array(pts)

def _local_optimize(points, max_iter=500):
    """Local optimization using Nelder-Mead."""
    x0 = points.flatten()
    
    res = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    if _is_valid(res.x.reshape(-1, 2)):
        return res.x.reshape(-1, 2)
    return points

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    h = np.sqrt(3) / 2
    
    best_points = None
    best_area = 0.0
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Hexagonal lattice
    starts.append(_hex_initial())
    
    # 2. Random starts
    for seed in [1, 7, 13, 42, 100, 200]:
        starts.append(_random_initial(seed))
    
    # 3. Perturbed hexagonal
    hex_pts = _hex_initial()
    for seed in [3, 5, 11]:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 0.03, size=hex_pts.shape)
        perturbed = hex_pts + noise
        perturbed = np.array([_project_to_triangle(p) for p in perturbed])
        starts.append(perturbed)
    
    for start in starts:
        if not _is_valid(start):
            start = np.array([_project_to_triangle(p) for p in start])
        
        optimized = _local_optimize(start, max_iter=400)
        
        if _is_valid(optimized):
            area = _min_triangle_area(optimized)
            if area > best_area:
                best_area = area
                best_points = optimized
    
    if best_points is None:
        best_points = _hex_initial()
    
    return best_points
# EVOLVE-BLOCK-END
