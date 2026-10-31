"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y < -tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _get_delaunay_critical_triangles(points, k=10):
    """
    Get the k smallest area triangles from the Delaunay triangulation.
    Returns list of (i, j, k) indices for the critical triangles.
    """
    try:
        tri = Delaunay(points)
        simplices = tri.simplices
        areas = []
        for simplex in simplices:
            i, j, k = simplex
            p1, p2, p3 = points[i], points[j], points[k]
            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            areas.append((area, (i, j, k)))
        
        # Sort by area and return the smallest k
        areas.sort(key=lambda x: x[0])
        return [idx for _, idx in areas[:k]]
    except Exception:
        # Fallback: return empty list if Delaunay fails
        return []

def _delaunay_constrained_objective(params, critical_indices):
    """
    Objective that focuses on maximizing the minimum area among the critical
    Delaunay triangles, with a penalty if any triangle becomes too small.
    """
    points = params.reshape(-1, 2)
    h = np.sqrt(3) / 2
    
    # Check if all points are inside the triangle
    if not _all_inside(points, tol=1e-4):
        penalty = 0.0
        for p in points:
            x, y = p[0], p[1]
            if x < 0 or x > 1 or y < 0 or y > h:
                penalty += 5.0
            if y > 2*h*x + 1e-3:
                penalty += 5.0
            if y > 2*h*(1-x) + 1e-3:
                penalty += 5.0
        return 1.0 + penalty
    
    # Compute areas of critical triangles
    min_critical_area = float('inf')
    for idx in critical_indices:
        i, j, k = idx
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < min_critical_area:
            min_critical_area = area
    
    # Also check global minimum to ensure no triangle is too small
    global_min = _min_triangle_area(points)
    
    # Objective: maximize the minimum of (critical min, global min)
    # Use a weighted combination to prioritize the critical triangles
    effective_min = 0.7 * min_critical_area + 0.3 * global_min
    return -effective_min

def _local_polish_delaunay(points, deadline):
    """
    Local refinement using Delaunay-informed coordinate descent.
    Focuses moves on points that are part of the smallest triangles.
    """
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    h = np.sqrt(3) / 2
    
    for pass_num in range(5):
        if time.time() > deadline - 5:
            break
        
        # Get critical triangles
        critical_indices = _get_delaunay_critical_triangles(points, k=15)
        
        # Identify which points are in critical triangles
        critical_points = set()
        for idx in critical_indices:
            for pt_idx in idx:
                critical_points.add(pt_idx)
        
        # Coordinate descent focusing on critical points first
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Process critical points first, then others
            point_order = sorted(critical_points) + [i for i in range(N) if i not in critical_points]
            
            for i in point_order:
                for coord in range(2):
                    for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                        old_val = points[i, coord]
                        
                        # Try positive step
                        points[i, coord] = old_val + step
                        if _all_inside(points, tol=1e-4):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area + 1e-10:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val
                        
                        # Try negative step
                        points[i, coord] = old_val - step
                        if _all_inside(points, tol=1e-4):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area + 1e-10:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val
            
            if not improved:
                break
        
        # Nelder-Mead polish on the full set
        if time.time() > deadline - 3:
            break
        
        def objective(params):
            pts = params.reshape(-1, 2)
            if not _all_inside(pts, tol=1e-4):
                return 10.0
            return -_min_triangle_area(pts)
        
        try:
            result = minimize(objective, points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _all_inside(new_points, tol=1e-4):
                    new_min_area = _min_triangle_area(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
    
    return points, current_min_area

def _build_structured_config():
    """
    Build a structured 11-point configuration based on known Heilbronn-type patterns.
    Uses a combination of vertices, edge midpoints, and interior points.
    """
    h = np.sqrt(3) / 2
    points = []
    
    # 3 vertices of the triangle (slightly inset to avoid boundary issues)
    eps = 0.02
    points.append([eps, eps])
    points.append([1-eps, eps])
    points.append([0.5, h - eps*2])
    
    # 3 edge midpoints
    points.append([0.5, eps])
    points.append([1-eps, h/2])
    points.append([eps, h/2])
    
    # Center point
    points.append([0.5, h/3])
    
    # 4 additional interior points in a hexagonal pattern
    cx, cy = 0.5, h/3
    r = 0.15
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi])
    for a in angles:
        px = cx + r*np.cos(a)
        py = cy + r*np.sin(a)*0.9
        if _inside_triangle([px, py], tol=1e-3):
            points.append([px, py])
    
    return np.array(points[:N])

def _optimize_with_delaunay_guidance(deadline):
    """
    Use Delaunay triangulation to guide the optimization.
    Start with a structured config, then iteratively refine.
    """
    h = np.sqrt(3) / 2
    
    # Start with structured configuration
    points = _build_structured_config()
    current_min_area = _min_triangle_area(points)
    
    # Iterative refinement with Delaunay guidance
    for iteration in range(10):
        if time.time() > deadline - 10:
            break
        
        # Get critical triangles
        critical_indices = _get_delaunay_critical_triangles(points, k=15)
        
        # Use Delaunay-informed local polish
        polished, min_area = _local_polish_delaunay(points, deadline)
        
        if min_area > current_min_area:
            points = polished
            current_min_area = min_area
        else:
            break
    
    return points, current_min_area

def _multi_start_delaunay(deadline):
    """
    Multiple starts with Delaunay-informed optimization.
    """
    h = np.sqrt(3) / 2
    best_points = None
    best_min_area = 0.0
    
    rng = np.random.RandomState(42)
    
    for seed in range(6):
        if time.time() > deadline - 15:
            break
        
        # Generate random points inside triangle
        points = []
        for _ in range(N):
            for _attempt in range(100):
                u, v = rng.random(), rng.random()
                if u + v < 0.9 and u > 0.05 and v > 0.05:
                    px = u + 0.5 * v
                    py = h * v
                    points.append([px, py])
                    break
        
        if len(points) < N:
            continue
        
        points = np.array(points)
        
        # Polish with Delaunay guidance
        polished, min_area = _local_polish_delaunay(points, deadline)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = polished
    
    return best_points, best_min_area

def construct():
    deadline = time.time() + 135  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Delaunay-guided optimization from structured start
    if time.time() < deadline - 40:
        points, min_area = _optimize_with_delaunay_guidance(deadline)
        if points is not None and min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
    
    # Strategy 2: Multi-start with Delaunay guidance
    if best_points is None or time.time() < deadline - 30:
        points, min_area = _multi_start_delaunay(deadline)
        if points is not None and min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
    
    # Strategy 3: Differential evolution with Delaunay-informed objective
    if best_points is None or time.time() < deadline - 20:
        # Start from best point or structured config
        if best_points is not None:
            initial_points = best_points.copy()
        else:
            initial_points = _build_structured_config()
        
        # Get critical triangles for the initial configuration
        critical_indices = _get_delaunay_critical_triangles(initial_points, k=10)
        
        bounds = [(0, 1)] * 2 * N
        
        try:
            de_result = differential_evolution(
                lambda params: _delaunay_constrained_objective(params, critical_indices),
                bounds,
                seed=42, maxiter=80, popsize=12,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _all_inside(de_points, tol=1e-4):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish_delaunay(best_points, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
