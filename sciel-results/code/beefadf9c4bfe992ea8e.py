"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.spatial import Delaunay, ConvexHull
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
    h = np.sqrt(3) / 2
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, h])
    
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

def _get_delaunay_triangles(points):
    """Get Delaunay triangulation and extract all triangles."""
    try:
        tri = Delaunay(points)
        triangles = []
        for simplex in tri.simplices:
            triangles.append(simplex)
        return triangles
    except Exception:
        return []

def _circumradius(p1, p2, p3):
    """Compute the circumradius of a triangle."""
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    
    area = _triangle_area(p1, p2, p3)
    if area < 1e-15:
        return float('inf')
    
    return (a * b * c) / (4.0 * area)

def _min_circumradius(points):
    """Compute the minimum circumradius among all Delaunay triangles."""
    triangles = _get_delaunay_triangles(points)
    if not triangles:
        return 0.0
    
    min_r = float('inf')
    for tri in triangles:
        r = _circumradius(points[tri[0]], points[tri[1]], points[tri[2]])
        if r < min_r:
            min_r = r
    return min_r

def _relaxation_objective(params, points_init):
    """
    Objective for convex relaxation: minimize negative log of min circumradius.
    params: [delta_x_0, delta_y_0, ..., delta_x_N-1, delta_y_N-1, t]
    where t is the target minimum circumradius.
    """
    n = len(points_init)
    deltas = params[:2*n].reshape(-1, 2)
    t = params[2*n]
    
    # Update points
    points = points_init + deltas
    
    # Check validity
    if not _is_valid(points):
        return 1e10
    
    # Compute all triangle areas and circumradii
    min_area = _min_triangle_area(points)
    
    # For Delaunay triangles, compute circumradii
    triangles = _get_delaunay_triangles(points)
    if not triangles:
        return 1e10
    
    min_r = float('inf')
    for tri in triangles:
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        r = _circumradius(p1, p2, p3)
        if r < min_r:
            min_r = r
    
    # We want to maximize min_r, so minimize -min_r + penalty if min_r < t
    penalty = 0.0
    if min_r < t:
        penalty = 100.0 * (t - min_r) ** 2
    
    return -min_r + penalty

def _construct_via_relaxation():
    """
    Construct points using Delaunay-based convex relaxation.
    """
    h = np.sqrt(3) / 2
    
    # Start with a good initial configuration
    # Use a symmetric pattern
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    best_points = None
    best_score = -1
    
    # Configuration 1: Random
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                pts.append([u + 0.5 * v, h * v])
        pts = np.array(pts)
        if _is_valid(pts):
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    
    # Configuration 2: Grid-based with symmetry
    # Place points in a triangular grid pattern
    for scale in range(2, 5):
        pts = []
        for i in range(scale + 1):
            for j in range(scale + 1 - i):
                u = i / scale
                v = j / scale
                if u + v < 1.0:
                    pts.append([u + 0.5 * v, h * v])
        if len(pts) >= N:
            pts = np.array(pts[:N])
            if _is_valid(pts):
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
    
    if best_points is None:
        # Fallback
        rng = np.random.default_rng(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98:
                pts.append([u + 0.5 * v, h * v])
        best_points = np.array(pts)
    
    # Now optimize using the relaxation approach
    points_init = best_points.copy()
    
    # We'll directly optimize coordinates but use Delaunay structure for guidance
    def _objective_direct(points_flat):
        points = points_flat.reshape(-1, 2)
        if not _is_valid(points):
            return 1e10
        
        min_area = _min_triangle_area(points)
        
        # Also consider circumradii of Delaunay triangles for better conditioning
        triangles = _get_delaunay_triangles(points)
        if triangles:
            min_r = float('inf')
            for tri in triangles:
                p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
                r = _circumradius(p1, p2, p3)
                if r < min_r:
                    min_r = r
            # Combine: we want both large min_area and large min_r
            # Use harmonic mean or similar
            combined = 2.0 * min_area * min_r / (min_area + min_r + 1e-10)
            return -combined
        else:
            return -min_area
    
    # Multi-start local optimization
    best_final_points = best_points.copy()
    best_final_score = _min_triangle_area(best_points)
    
    for start_seed in range(5):
        if start_seed == 0:
            x0 = best_points.flatten()
        else:
            # Perturb initial points
            rng = np.random.default_rng(start_seed + 100)
            x0 = best_points.flatten() + rng.normal(0, 0.05, size=2*N)
        
        res = minimize(
            _objective_direct,
            x0,
            method='BFGS',
            options={'maxiter': 2000, 'gtol': 1e-12}
        )
        
        if res.success:
            candidate = res.x.reshape(-1, 2)
            if _is_valid(candidate):
                score = _min_triangle_area(candidate)
                if score > best_final_score:
                    best_final_score = score
                    best_final_points = candidate
    
    # Final refinement with differential evolution on a smaller neighborhood
    bounds = []
    for i in range(N):
        bounds.append((max(0, best_final_points[i][0] - 0.1), min(1, best_final_points[i][0] + 0.1)))
        bounds.append((max(0, best_final_points[i][1] - 0.1), min(h, best_final_points[i][1] + 0.1)))
    
    result = differential_evolution(
        _objective_direct,
        bounds=bounds,
        seed=42,
        maxiter=50,
        tol=1e-10,
        polish=True,
        init='sobol'
    )
    
    if result.success and _is_valid(result.x.reshape(-1, 2)):
        candidate = result.x.reshape(-1, 2)
        score = _min_triangle_area(candidate)
        if score > best_final_score:
            best_final_points = candidate
    
    return best_final_points

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    points = _construct_via_relaxation()
    
    # Final validation
    if not _is_valid(points):
        h = np.sqrt(3) / 2
        rng = np.random.default_rng(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98:
                pts.append([u + 0.5 * v, h * v])
        points = np.array(pts)
    
    return points
# EVOLVE-BLOCK-END
