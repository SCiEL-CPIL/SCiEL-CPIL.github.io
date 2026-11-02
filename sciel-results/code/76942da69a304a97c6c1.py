"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
# Area of equilateral triangle with side 1: sqrt(3)/4
TRI_AREA = np.sqrt(3) / 4.0

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by points p1, p2, p3 (each shape (2,))."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def _min_triangle_area(points):
    """Minimum area over all C(11,3)=165 triangles."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the reference triangle."""
    # Reference triangle: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def _penalty(points):
    """Penalty for points outside the triangle (should be 0 for valid configs)."""
    pen = 0.0
    for p in points:
        if not _is_inside_triangle(p):
            # Distance to triangle boundary (approximate)
            pen += 10.0
    return pen

def _objective_flat(flat):
    """Objective for differential evolution: minimize negative min area + penalty."""
    pts = flat.reshape(-1, 2)
    if _penalty(pts) > 0:
        return 100.0
    min_a = _min_triangle_area(pts)
    return -min_a

def _construct_single_start(start_points, max_iter=500, tol=1e-10):
    """
    Given an initial configuration of 11 points (flattened to 22 dims),
    run a local optimization to maximize the minimum triangle area.
    """
    pts = start_points.reshape(-1, 2)
    
    # Verify all points are inside
    for p in pts:
        if not _is_inside_triangle(p):
            return None
    
    def neg_min_area(flat):
        p = flat.reshape(-1, 2)
        # Check bounds
        for pp in p:
            if not _is_inside_triangle(pp, eps=-1e-8):
                return 1e6
        return -_min_triangle_area(p)
    
    # Use Nelder-Mead for local refinement
    try:
        result = minimize(
            neg_min_area,
            flat,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        best_pts = result.x.reshape(-1, 2)
        # Verify validity
        valid = all(_is_inside_triangle(bp) for bp in best_pts)
        if not valid:
            return None
        min_a = _min_triangle_area(best_pts)
        return best_pts, min_a
    except Exception:
        return None

def _random_inside_triangle(rng, n_points):
    """Generate n_points randomly inside the unit equilateral triangle."""
    pts = []
    for _ in range(n_points):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                p = np.array([u + 0.5*v, (np.sqrt(3)/2)*v])
                pts.append(p)
                break
    return np.array(pts)

def _grid_initialization():
    """
    Create a structured initial configuration using a grid-like pattern
    inside the triangle to provide a good starting point.
    """
    pts = []
    # Add vertices
    pts.append(np.array([0.0, 0.0]))
    pts.append(np.array([1.0, 0.0]))
    pts.append(np.array([0.5, np.sqrt(3)/2]))
    
    # Add midpoints of edges
    pts.append(np.array([0.5, 0.0]))
    pts.append(np.array([0.75, np.sqrt(3)/4]))
    pts.append(np.array([0.25, np.sqrt(3)/4]))
    
    # Add centroid
    pts.append(np.array([0.5, np.sqrt(3)/6]))
    
    # Add a few more interior points
    pts.append(np.array([0.25, np.sqrt(3)/12]))
    pts.append(np.array([0.75, np.sqrt(3)/12]))
    pts.append(np.array([0.375, np.sqrt(3)/6]))
    pts.append(np.array([0.625, np.sqrt(3)/6]))
    
    return np.array(pts[:N])

def construct():
    """
    Place 11 points on or inside the unit equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    Returns an array of shape (11, 2).
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    best_pts = None
    best_min_area = 0.0
    
    # Strategy 1: Start from a structured grid and refine
    initial = _grid_initialization()
    result = _construct_single_start(initial.flatten(), max_iter=300)
    if result is not None:
        pts, min_a = result
        if min_a > best_min_area:
            best_min_area = min_a
            best_pts = pts
    
    # Check deadline
    if time.time() > deadline:
        if best_pts is not None:
            return best_pts
        return initial
    
    # Strategy 2: Multiple random starts with local refinement
    import random
    rng = random.Random(42)
    
    num_random_starts = 5
    for i in range(num_random_starts):
        if time.time() > deadline:
            break
        
        random_pts = _random_inside_triangle(rng, N)
        result = _construct_single_start(random_pts.flatten(), max_iter=200)
        if result is not None:
            pts, min_a = result
            if min_a > best_min_area:
                best_min_area = min_a
                best_pts = pts
    
    # Strategy 3: Differential evolution for global search (if time permits)
    if time.time() < deadline - 60.0:
        try:
            bounds = []
            for _ in range(N):
                bounds.append([(0.0, 1.0), (0.0, np.sqrt(3)/2)])
            
            def de_objective(flat):
                pts = flat.reshape(-1, 2)
                # Quick check: all x in [0,1], y in [0, sqrt(3)/2]
                if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] > 1) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > np.sqrt(3)/2):
                    return 100.0
                # Check inside triangle
                for p in pts:
                    if not _is_inside_triangle(p):
                        return 100.0
                return -_min_triangle_area(pts)
            
            de_result = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                polish=False
            )
            
            if de_result.fun < 100.0:
                de_pts = de_result.x.reshape(-1, 2)
                # Refine with local optimization
                result = _construct_single_start(de_result.x, max_iter=300)
                if result is not None:
                    pts, min_a = result
                    if min_a > best_min_area:
                        best_min_area = min_a
                        best_pts = pts
        except Exception:
            pass
    
    # Final check: return best found
    if best_pts is not None:
        return best_pts
    
    # Fallback: return the grid initialization
    return initial

# EVOLVE-BLOCK-END
