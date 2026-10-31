"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

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
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, h])
    
    # Using barycentric coordinates
    # For point P, compute barycentric coordinates relative to ABC
    v0 = B - A
    v1 = C - A
    v2 = p - A
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    
    return u >= -tol and v >= -tol and (u + v) <= 1 + tol

def _is_valid(points):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p) for p in points)

def _objective(points_flat):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = points_flat.reshape(-1, 2)
    if not _is_valid(points):
        return 1e10  # Large penalty for invalid points
    
    min_area = _min_triangle_area(points)
    return -min_area

def _initial_guess():
    """Generate an initial guess for the points."""
    # Start with a grid-like arrangement inside the triangle
    # Use a simple heuristic: place points in a pattern that spreads them out
    h = np.sqrt(3) / 2
    points = []
    
    # Try a few initial patterns and pick the best
    best_points = None
    best_area = -1
    
    # Pattern 1: Random with fixed seed
    import random
    rng = random.Random(42)
    for _ in range(100):
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.05 and v > 0.05:
                pts.append([u + 0.5 * v, h * v])
        if _is_valid(pts):
            area = _min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts
    
    # Pattern 2: Grid-based
    # Divide the triangle into smaller regions and place points
    for scale in [2, 3, 4]:
        for offset_u in np.linspace(0.1, 0.5, scale + 1):
            for offset_v in np.linspace(0.1, 0.5, scale + 1):
                pts = []
                for i in range(scale + 1):
                    for j in range(scale + 1):
                        if i + j <= scale:
                            u = offset_u * i / scale
                            v = offset_v * j / scale
                            if u + v < 0.95:
                                pts.append([u + 0.5 * v, h * v])
                if len(pts) >= N:
                    pts = pts[:N]
                    if _is_valid(pts):
                        area = _min_triangle_area(pts)
                        if area > best_area:
                            best_area = area
                            best_points = pts
    
    if best_points is None:
        # Fallback to the original random method
        import random
        rng = random.Random(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, h * v])
        best_points = pts
    
    return np.array(best_points).flatten()

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    h = np.sqrt(3) / 2
    
    # Get initial guess
    x0 = _initial_guess()
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, h]
    bounds = [(0, 1), (0, h)] * N
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        _objective,
        bounds=bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=False,
        init='sobol'
    )
    
    best_points = result.x.reshape(-1, 2)
    
    # Refine with local optimization
    if _is_valid(best_points):
        # Use BFGS for local refinement
        res = minimize(
            _objective,
            best_points.flatten(),
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-10}
        )
        
        if res.success and _is_valid(res.x.reshape(-1, 2)):
            best_points = res.x.reshape(-1, 2)
    
    # Final validation
    if not _is_valid(best_points):
        # Fallback to a valid configuration
        import random
        rng = random.Random(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, h * v])
        best_points = np.array(pts)
    
    return best_points
# EVOLVE-BLOCK-END
