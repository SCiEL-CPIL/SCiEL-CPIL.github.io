"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# Deadline guard: kill after 140s to be safe (limit is 150s)
_DEADLINE = 140.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This equals the largest circumradius of the facets
    of the convex hull (in angular terms).
    
    For a set of points on the unit sphere, the covering radius is the maximum over
    all facets (triangles) of the convex hull of the angular circumradius of that
    triangle.
    
    We compute this by:
    1. Computing the convex hull
    2. For each facet (triangle), computing the circumcenter
    3. The angular distance from the circumcenter to any vertex is the circumradius
    """
    points = np.asarray(points)
    points = np.array([_normalize(p) for p in points])
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        return _covering_radius_fallback(points)
    
    max_radius = 0.0
    
    # For each facet, compute the circumradius
    for simplex in hull.simplices:
        # simplex has 3 vertices (triangles in 3D)
        v0 = points[simplex[0]]
        v1 = points[simplex[1]]
        v2 = points[simplex[2]]
        
        # Compute circumcenter of the triangle on the sphere
        # The circumcenter of a spherical triangle is the point equidistant from all three vertices
        # We can find it as the intersection of the plane through the three points with the sphere
        # (normalized to unit length)
        
        # The circumcenter in 3D space for a triangle on the unit sphere:
        # It's the normalized vector that is equidistant from all three vertices
        
        # Method: solve for a point c such that |c - v0| = |c - v1| = |c - v2| and |c| = 1
        # Since all vertices are on the unit sphere, the circumcenter lies on the line
        # through the origin and the centroid of the triangle (in the plane of the triangle)
        
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
        # is the normalized vector perpendicular to the plane containing the three points,
        # but that's only for right triangles. Let me think more carefully.
        
        # The covering radius is the maximum over all points y on the sphere of the
        # minimum angular distance to any point in the set.
        
        # For a convex hull facet (triangle), the farthest point from all three vertices
        # within that facet's cap is the circumcenter of the triangle (if the circumcenter
        # is inside the spherical triangle) or one of the vertices.
        
        # The angular circumradius of a spherical triangle with vertices a, b, c is:
        # the angular distance from the circumcenter to each vertex.
        
        # To find the circumcenter: it's the point on the sphere equidistant from a, b, c.
        # This means it's in the plane perpendicular to (a-b), (a-c) passing through... 
        # Actually, it's the intersection of the sphere with the line that is the 
        # circumcenter of the planar triangle formed by projecting to the tangent plane,
        # but that's complex.
        
        # Simpler approach: the circumcenter c satisfies:
        # c · a = c · b = c · c' (where c' is the circumradius cosine)
        # and |c| = 1
        
        # c · a = c · b => c · (a - b) = 0
        # c · a = c · c => c · (a - c) = 0
        # So c is perpendicular to (a-b) and (a-c), meaning c is parallel to (a-b) × (a-c)
        
        n = np.cross(a - b, a - c) if False else np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            # Degenerate triangle
            continue
        
        n = n / n_norm
        
        # The circumcenter is either n or -n (whichever is on the same side as the triangle)
        # Check which sign gives a point that is "inside" the cap
        c1 = n
        c2 = -n
        
        # The circumcenter should be in the direction that makes the angular distances
        # to all three vertices equal and minimal
        # Compute angular distance from c1 to a
        d1 = np.arccos(np.clip(np.dot(c1, a), -1, 1))
        d2 = np.arccos(np.clip(np.dot(c2, a), -1, 1))
        
        # The circumradius is the angular distance from circumcenter to any vertex
        # We want the smaller one (the circumcenter that's on the same hemisphere)
        if d1 < d2:
            circumcenter = c1
            circumradius = d1
        else:
            circumcenter = c2
            circumradius = d2
        
        # Verify: check that the angular distances to all three vertices are approximately equal
        d0 = np.arccos(np.clip(np.dot(circumcenter, a), -1, 1))
        d1v = np.arccos(np.clip(np.dot(circumcenter, b), -1, 1))
        d2v = np.arccos(np.clip(np.dot(circumcenter, c), -1, 1))
        
        # The circumradius should be max of these (they should be approximately equal)
        circumradius = max(d0, d1v, d2v)
        
        # But we also need to check that the circumcenter is actually inside the spherical triangle
        # If not, the maximum distance might be at a vertex or edge midpoint
        # For now, use the circumradius as an upper bound
        
        max_radius = max(max_radius, circumradius)
    
    return max_radius


def _covering_radius_fallback(points):
    """Fallback covering radius using a dense grid."""
    points = np.asarray(points)
    points = np.array([_normalize(p) for p in points])
    
    # Create a grid of test points
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    max_dist = 0.0
    for t in theta:
        for p in phi:
            y = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            # Find nearest point
            dists = np.arccos(np.clip(points @ y, -1, 1))
            max_dist = max(max_dist, np.min(dists))
    
    return max_dist


def _objective(x, points_fixed=None):
    """Objective function: covering radius of the configuration."""
    # x is a flat array of 3*N values
    pts = x.reshape(-1, 3)
    pts = np.array([_normalize(p) for p in pts])
    return _covering_radius(pts)


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _simulated_annealing(points, max_iter=5000, T0=0.1, T_end=0.001, cooling=0.999):
    """Simulated annealing to minimize covering radius."""
    points = np.array([_normalize(p) for p in points])
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    current_radius = best_radius
    
    T = T0
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Generate a perturbation
        idx = rng.integers(0, len(points))
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = T * 0.5
        # Perturb in a random direction on the tangent plane
        tangent = rng.standard_normal(3)
        tangent -= np.dot(tangent, points[idx]) * points[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            continue
        tangent = tangent / tangent_norm
        
        new_point = points[idx] + magnitude * tangent
        new_point = _normalize(new_point)
        
        new_points = points.copy()
        new_points[idx] = new_point
        
        new_radius = _covering_radius(new_points)
        delta = new_radius - current_radius
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            points = new_points
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        T *= cooling
        T = max(T, T_end)
    
    return best_points, best_radius


def _local_optimization(points, max_iter=2000):
    """Local optimization using gradient-based methods on the covering radius."""
    points = np.array([_normalize(p) for p in points])
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Use Nelder-Mead or BFGS on the flat representation
    # This is slow, so we use a simpler approach: iterate over points and try to move each
    # to reduce the covering radius
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        improved = False
        for i in range(len(points)):
            if _time_left() < 5.0:
                break
            
            # Try moving point i in a small random direction
            for _ in range(5):
                tangent = np.random.standard_normal(3)
                tangent -= np.dot(tangent, points[i]) * points[i]
                tn = np.linalg.norm(tangent)
                if tn < 1e-10:
                    continue
                tangent = tangent / tn
                
                for step in [0.01, 0.005, 0.001]:
                    new_point = _normalize(points[i] + step * tangent)
                    new_points = points.copy()
                    new_points[i] = new_point
                    
                    new_radius = _covering_radius(new_points)
                    if new_radius < best_radius - 1e-10:
                        points = new_points
                        best_points = points.copy()
                        best_radius = new_radius
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_points, best_radius


def construct():
    """
    Construct 50 points on the unit sphere to minimize the covering radius.
    Uses a combination of Fibonacci spiral initialization, simulated annealing,
    and local optimization with multiple restarts.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Fibonacci spiral
            points = _fibonacci_sphere(N)
        elif restart == 1:
            # Fibonacci with offset
            points = _fibonacci_sphere(N, seed_offset=0.3)
        elif restart == 2:
            # Random initialization
            rng = np.random.default_rng(restart)
            points = rng.standard_normal((N, 3))
            points = np.array([_normalize(p) for p in points])
        elif restart == 3:
            # Fibonacci with different offset
            points = _fibonacci_sphere(N, seed_offset=1.0)
        else:
            # Another random
            rng = np.random.default_rng(restart + 10)
            points = rng.standard_normal((N, 3))
            points = np.array([_normalize(p) for p in points])
        
        # Simulated annealing
        points, radius = _simulated_annealing(points, max_iter=3000, T0=0.05, T_end=0.001, cooling=0.9995)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Local optimization
        points, radius = _local_optimization(points, max_iter=500)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        if _time_left() < 10.0:
            break
    
    # Final check
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Ensure all points are on the unit sphere
    best_points = np.array([_normalize(p) for p in best_points])
    
    return best_points

# EVOLVE-BLOCK-END
