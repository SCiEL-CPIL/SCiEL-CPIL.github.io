"""Improved covering-radius optimizer for 50 points on the unit sphere."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the
    sphere to its nearest chosen point. This equals the maximum circumradius
    of the facets of the convex hull (for facets whose circumcenter lies on
    the sphere side, i.e., the cap is empty).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute-force via dense sampling
        return _covering_radius_brute(pts)
    
    # Get unique vertices
    verts = np.unique(hull.vertices)
    pts_v = pts[verts]
    
    max_cos = 1.0  # cos of covering radius (we want to minimize angle, maximize cos)
    
    for simplex in hull.simplices:
        # Get the 3 points defining this facet
        tri = pts[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter of the triangle on the sphere
        # The circumcenter in 3D is the point equidistant from a, b, c
        # For points on the unit sphere, the circumcenter of the spherical triangle
        # is along the direction (a x b) x c or similar... 
        # Actually, for a spherical triangle with vertices on unit sphere,
        # the circumcenter direction is proportional to (a+b+c) only if it's
        # an equilateral triangle. In general, we need the point equidistant from all three.
        
        # The circumcenter of three points on the unit sphere:
        # Find point p such that |p-a|^2 = |p-b|^2 = |p-c|^2 and |p|=1
        # |p-a|^2 = 2 - 2 p·a, so we need p·a = p·b = p·c
        # This means p is orthogonal to (a-b) and (a-c)
        # So p is along (a-b) x (a-c)
        
        d1 = a - b
        d2 = a - c
        cross = np.cross(d1, d2)
        norm = np.linalg.norm(cross)
        
        if norm < 1e-12:
            continue
        
        p = cross / norm
        
        # Check which side: we want the cap that doesn't contain other points
        # The two candidates are p and -p. The correct one is the one
        # where the cap (points closer to p than to any other config point)
        # is the "empty" one. For the convex hull facet, the outward normal
        # points away from the origin.
        
        # Actually, for the covering radius, we need the farthest point on the
        # sphere from all config points. The candidates are the circumcenters
        # of the triangular facets. For each facet, the circumcenter that
        # is on the "outside" (same side as the outward normal of the hull facet)
        # gives the vertex of the Voronoi cell that could be the farthest.
        
        # The outward normal of the hull facet is the normalized cross product
        # in the correct orientation. Let's compute both p and -p and pick
        # the one that has positive dot product with the centroid of the triangle
        # (which points outward from origin for a convex hull containing origin).
        
        centroid = (a + b + c) / 3.0
        if np.dot(p, centroid) < 0:
            p = -p
        
        cos_val = np.dot(p, a)  # = np.dot(p, b) = np.dot(p, c)
        if cos_val > max_cos:
            max_cos = cos_val
    
    # Covering radius angle = arccos(max_cos)
    # But we need to be careful: the covering radius is the MAXIMUM distance,
    # which corresponds to the MINIMUM cos value among the Voronoi vertices.
    # Wait, I have it backwards. Let me reconsider.
    
    # The covering radius is the largest angular distance from any sphere point
    # to its nearest config point. The farthest points are the Voronoi vertices
    # (circumcenters of hull facets). For each such vertex, the distance to the
    # nearest config point is the distance to the 3 points of that facet.
    
    # So we want the MAXIMUM over all facets of arccos(p · a) where p is the
    # circumcenter. Since arccos is decreasing, this is arccos(MIN over facets of p·a).
    
    # Let me redo: track the minimum cos value
    min_cos = 1.0
    
    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        d1 = a - b
        d2 = a - c
        cross = np.cross(d1, d2)
        norm = np.linalg.norm(cross)
        
        if norm < 1e-12:
            continue
        
        p = cross / norm
        
        centroid = (a + b + c) / 3.0
        if np.dot(p, centroid) < 0:
            p = -p
        
        cos_val = np.dot(p, a)
        if cos_val < min_cos:
            min_cos = cos_val
    
    # Clamp for numerical safety
    min_cos = np.clip(min_cos, -1.0, 1.0)
    return np.degrees(np.arccos(min_cos))


def _covering_radius_brute(pts, n_samples=50000):
    """Brute-force estimate of covering radius by sampling the sphere."""
    # Use Fibonacci lattice for sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find distance to nearest point
    # Use matrix multiplication for speed
    # dist^2 = 2 - 2 * dot(sample, pts)
    dots = samples @ pts.T  # (n_samples, N)
    max_dots = dots.max(axis=1)
    min_cos = max_dots.min()
    min_cos = np.clip(min_cos, -1.0, 1.0)
    return np.degrees(np.arccos(min_cos))


def _objective(params):
    """Compute the objective (covering radius in degrees) from parameters.
    
    params: (2N,) array of spherical coordinates (theta, phi) for each point.
    theta in [0, pi], phi in [0, 2*pi].
    """
    theta = params[0::2]
    phi = params[1::2]
    
    # Convert to Cartesian
    sin_theta = np.sin(theta)
    pts = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        np.cos(theta)
    ])
    
    return _covering_radius(pts)


def _grad_objective(params, eps=1e-7):
    """Numerical gradient of the objective."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _random_params():
    """Generate random spherical coordinates."""
    theta = np.random.uniform(0, np.pi, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi
    return params


def _fibonacci_params():
    """Generate Fibonacci spiral parameters as initial guess."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    z = np.clip(z, -1.0 + 1e-10, 1.0 - 1e-10)
    theta = np.arccos(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi
    return params


def _local_search(params, max_iters=200, tol=1e-8, time_limit=None):
    """Gradient descent with adaptive step size on spherical coordinates."""
    start_time = time.time()
    current = params.copy()
    current_obj = _objective(current)
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute gradient
        grad = _grad_objective(current)
        
        # Adaptive step size: start with moderate step, reduce if objective increases
        step = 0.01
        improved = False
        for trial in range(20):
            new_params = current - step * grad
            new_obj = _objective(new_params)
            
            if new_obj < current_obj:
                current = new_params
                current_obj = new_obj
                improved = True
                break
            else:
                step *= 0.5
        
        if not improved:
            break
        
        if abs(grad).max() < tol:
            break
    
    return current, current_obj


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start with Fibonacci spiral
    best_params = _fibonacci_params()
    best_obj = _objective(best_params)
    
    # Do local search from the Fibonacci start
    remaining = deadline - time.time()
    if remaining > 5:
        best_params, best_obj = _local_search(best_params, max_iters=300, time_limit=remaining - 5)
    
    # Try a few random restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        random_params = _random_params()
        # Quick local search with fewer iterations
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        local_params, local_obj = _local_search(random_params, max_iters=100, time_limit=remaining)
        
        if local_obj < best_obj:
            best_params = local_params
            best_obj = local_obj
    
    # Final polish if time permits
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            best_params, best_obj = _local_search(best_params, max_iters=200, time_limit=remaining)
    
    # Convert best parameters to Cartesian points
    theta = best_params[0::2]
    phi = best_params[1::2]
    sin_theta = np.sin(theta)
    pts = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        np.cos(theta)
    ])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    return pts
