"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _covering_radius(points):
    """Compute the covering radius in degrees from the convex hull."""
    try:
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            p = points[simplex]
            # Compute circumcenter of the simplex
            # For a triangle on the sphere, circumradius = angle from center to circumcenter
            # Use the formula: for points on unit sphere, circumradius of a facet
            # is the angle from the origin to the plane's circumcenter
            # Simpler: compute the circumradius of the spherical triangle
            # The covering radius is the maximum over all facets of the circumradius
            # of that facet (which is the angular distance from the facet center to its vertices)
            
            # For a triangle with vertices a, b, c on unit sphere:
            # The circumcenter is the point equidistant from all three
            # The circumradius is the angular distance from circumcenter to any vertex
            
            # Method: find the normal to the plane, then find circumcenter
            a, b, c = p[0], p[1], p[2]
            
            # Normal to the plane
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-10:
                continue
            n = n / nn
            
            # The circumcenter lies on the line from origin through n (or -n)
            # For a spherical triangle, the circumcenter is at distance 1/|n| along n
            # if |n| > 0, but we need to be careful
            # Actually, the circumcenter of the triangle in 3D space:
            # Let's compute it properly
            
            # Set up linear system: |x - a|^2 = |x - b|^2 = |x - c|^2
            # This gives: 2(b-a).x = |b|^2 - |a|^2 = 0 (since all on unit sphere)
            # 2(c-a).x = |c|^2 - |a|^2 = 0
            # So x is orthogonal to (b-a) and (c-a), meaning x is along n
            
            # The circumcenter in 3D is at x = t * n where t is determined by
            # |t*n - a|^2 = R^2 for some R
            # t^2 - 2t(n.a) + 1 = R^2
            # The circumcenter is the point on the line through origin in direction n
            # that is equidistant from a, b, c
            
            # Actually for points on unit sphere, the circumcenter of the triangle
            # (in the sense of the sphere) is the point on the sphere equidistant from a,b,c
            # This is the spherical circumcenter
            
            # The angular circumradius is the angle from this point to a (or b or c)
            
            # To find the spherical circumcenter: it's the intersection of the sphere
            # with the line through the origin in direction n (if the triangle is not
            # containing the antipode)
            
            # The spherical circumcenter is at angle alpha from a where:
            # cos(alpha) = n . a / |n| ... no, that's not right.
            
            # Let me use a different approach: the circumradius of a spherical triangle
            # with angles A, B, C and side lengths a, b, c is:
            # Actually, let's just compute it numerically.
            
            # The point on the sphere equidistant from a, b, c is along ±n
            # Check which sign gives the closest point
            n_pos = n
            n_neg = -n
            
            d_pos = np.arccos(np.clip(np.dot(n_pos, a), -1, 1))
            d_neg = np.arccos(np.clip(np.dot(n_neg, a), -1, 1))
            
            # The circumcenter is the one that is closer to all three vertices
            # (the one inside the triangle)
            if d_pos < d_neg:
                circ_center = n_pos
            else:
                circ_center = n_neg
            
            circumradius = np.arccos(np.clip(np.dot(circ_center, a), -1, 1))
            max_circumradius = max(max_circumradius, circumradius)
        
        return math.degrees(max_circumradius)
    except Exception:
        return 180.0


def _soft_max_covering_radius(points, grid, beta=50.0):
    """Soft-max surrogate for the covering radius."""
    # Compute angular distances from each grid point to nearest configuration point
    # grid: (M, 3), points: (N, 3)
    # dist[i,j] = arccos(grid[i] . points[j])
    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    
    # For each grid point, find the min distance to any configuration point
    min_dists = np.min(dists, axis=1)  # (M,)
    
    # Soft max of min_dists
    max_d = np.max(min_dists)
    # Shift for numerical stability
    shifted = min_dists - max_d
    weights = np.exp(beta * shifted)
    log_sum_exp = np.log(np.sum(weights)) / beta
    soft_max = max_d + log_sum_exp
    
    return math.degrees(soft_max)


def _generate_grid(n_lat=60, n_lon=120):
    """Generate a dense spherical grid."""
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    x = np.cos(LA) * np.cos(LO)
    y = np.cos(LA) * np.sin(LO)
    z = np.sin(LA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid


def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration
    points = _fibonacci_sphere(N).copy()
    
    # Generate grid for soft-max surrogate
    grid = _generate_grid(n_lat=80, n_lon=160)
    
    # First, do a local optimization with the soft-max surrogate
    def objective(params):
        pts = _params_to_points(params)
        return _soft_max_covering_radius(pts, grid, beta=30.0)
    
    def _params_to_points(params):
        # params: 2N parameters (theta, phi for each point)
        theta = params[0::2]
        phi = params[1::2]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack([x, y, z])
        return pts
    
    def _points_to_params(pts):
        theta = np.arccos(np.clip(pts[:, 2], -1, 1))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        return np.column_stack([theta, phi]).ravel()
    
    # Optimize using L-BFGS-B
    params0 = _points_to_params(points)
    
    result = minimize(objective, params0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success or result.fun < objective(params0):
        points = _params_to_points(result.x)
    
    # Check time
    if time.time() > deadline:
        return points
    
    # Now do targeted gradient descent on worst facets
    def _worst_facet_gradient(points):
        """Compute gradient for the worst facet."""
        try:
            hull = ConvexHull(points)
        except:
            return None, None
        
        # Find the facet with maximum circumradius
        max_cr = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            p = points[simplex]
            a, b, c = p[0], p[1], p[2]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-10:
                continue
            n = n / nn
            
            d_pos = np.arccos(np.clip(np.dot(n, a), -1, 1))
            d_neg = np.arccos(np.clip(np.dot(-n, a), -1, 1))
            
            if d_pos < d_neg:
                circ_center = n
            else:
                circ_center = -n
            
            cr = np.arccos(np.clip(np.dot(circ_center, a), -1, 1))
            if cr > max_cr:
                max_cr = cr
                worst_simplex = simplex
        
        if worst_simplex is None:
            return None, None
        
        # Gradient: push the vertices of the worst facet towards the circumcenter
        # to reduce the circumradius
        p = points[worst_simplex]
        a, b, c = p[0], p[1], p[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            return None, None
        n = n / nn
        
        d_pos = np.arccos(np.clip(np.dot(n, a), -1, 1))
        d_neg = np.arccos(np.clip(np.dot(-n, a), -1, 1))
        
        if d_pos < d_neg:
            circ_center = n
        else:
            circ_center = -n
        
        # Gradient direction: move each vertex towards circumcenter on the sphere
        grad = np.zeros_like(points)
        for idx, pt in zip(worst_simplex, p):
            # Tangent direction from pt towards circ_center
            tangent = circ_center - np.dot(circ_center, pt) * pt
            tn = np.linalg.norm(tangent)
            if tn > 1e-10:
                grad[idx] = tangent / tn
        
        return grad, worst_simplex
    
    # Targeted gradient descent with adaptive step size
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    step_size = 0.01
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        grad, simplex = _worst_facet_gradient(points)
        if grad is None:
            break
        
        # Take a step
        new_points = points + step_size * grad
        # Normalize
        new_points = np.array([_normalize(p) for p in new_points])
        
        # Check if it's valid (not in a hemisphere, etc.)
        cr = _covering_radius(new_points)
        if cr < best_cr:
            points = new_points
            best_points = points.copy()
            best_cr = cr
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.7
        
        # Random restart if stuck
        if iteration % 500 == 499 and step_size < 0.001:
            # Perturb randomly
            noise = np.random.randn(N, 3) * 0.1
            new_points = points + noise
            new_points = np.array([_normalize(p) for p in new_points])
            cr = _covering_radius(new_points)
            if cr < best_cr:
                points = new_points
                best_points = points.copy()
                best_cr = cr
            step_size = 0.01
    
    # Final polish with L-BFGS-B on the best points
    if time.time() < deadline - 5.0:
        params0 = _points_to_params(best_points)
        result2 = minimize(objective, params0, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        if result2.success:
            polished = _params_to_points(result2.x)
            cr_polished = _covering_radius(polished)
            if cr_polished < best_cr:
                best_points = polished
                best_cr = cr_polished
    
    return best_points
# EVOLVE-BLOCK-END
