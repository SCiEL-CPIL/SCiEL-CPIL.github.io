"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        # Normalize points
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms < 1e-12):
            return 180.0
        points = points / norms[:, np.newaxis]
        
        # Check if all points are in a closed hemisphere
        # A simple check: if the convex hull has a facet with outward normal
        # such that all points are on one side, we're in a hemisphere.
        # More robust: check if there exists a direction d such that d·p_i >= 0 for all i
        # This is equivalent to the origin being outside the convex hull.
        # We'll use the convex hull to compute the covering radius.
        
        hull = ConvexHull(points)
        
        # For each facet, compute the circumcenter of the facet (on the sphere)
        # and the angular radius of the cap that covers that facet.
        # The covering radius is the maximum of these.
        
        # Get facet normals and distances from origin
        # For a facet with vertices v1, v2, ..., vk, the center of the circumscribed cap
        # is the normalized sum of the vertices (if the facet is a simplex, this works;
        # for general facets, we need the circumcenter in 3D).
        
        # Actually, for the covering radius, we need the farthest point on the sphere
        # from all configuration points. This is the maximum over all facets of the
        # angular distance from the facet's circumcenter to its vertices, but only if
        # the circumcenter is on the correct side.
        
        # A more direct approach: the covering radius is the maximum over all facets
        # of the circumradius of that facet (the angular radius of the smallest cap
        # containing the facet vertices), provided the cap is "empty" (doesn't contain
        # any other points in its interior).
        
        # Let's compute the circumcenter for each facet.
        # For a triangular facet (which is what we'll have in general position),
        # the circumcenter in 3D can be computed.
        
        max_radius = 0.0
        for simplex in hull.simplices:
            # simplex has 3 vertices for a 3D convex hull
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute the circumcenter of the triangle v1, v2, v3 in 3D
            # The circumcenter is the point equidistant from v1, v2, v3
            # and lies in the plane of the triangle.
            
            # Using the formula for circumcenter of a triangle in 3D
            a = np.linalg.norm(v2 - v3)
            b = np.linalg.norm(v1 - v3)
            c = np.linalg.norm(v1 - v2)
            
            # Avoid division by zero
            if a < 1e-12 or b < 1e-12 or c < 1e-12:
                continue
            
            # Circumcenter formula
            # C = (a^2 * (|b|^2 + |c|^2 - |a|^2) * v1 + 
            #       b^2 * (|a|^2 + |c|^2 - |b|^2) * v2 + 
            #       c^2 * (|a|^2 + |b|^2 - |c|^2) * v3) / (2 * |cross(v1-v3, v2-v3)|^2)
            
            # Simpler: use the fact that for unit vectors, the circumcenter
            # of the spherical triangle is the normalized sum if it's acute.
            # But let's use the proper 3D circumcenter.
            
            # The circumcenter lies at:
            # C = v1 + t1*(v2-v1) + t2*(v3-v1)
            # where t1, t2 solve the linear system for equidistance.
            
            d1 = v2 - v1
            d2 = v3 - v1
            
            # C = v1 + t1*d1 + t2*d2
            # |C - v1|^2 = |C - v2|^2 => 2*t1*|d1|^2 + 2*t2*(d1·d2) = |d1|^2
            # |C - v1|^2 = |C - v3|^2 => 2*t1*(d1·d2) + 2*t2*|d2|^2 = |d2|^2
            
            A = np.array([[np.dot(d1, d1), np.dot(d1, d2)],
                          [np.dot(d1, d2), np.dot(d2, d2)]])
            b_vec = np.array([np.dot(d1, d1), np.dot(d2, d2)])
            
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) < 1e-15:
                continue
            
            t = np.linalg.solve(A, b_vec)
            C = v1 + t[0]*d1 + t[1]*d2
            
            # Normalize C to get the spherical circumcenter
            C_norm = np.linalg.norm(C)
            if C_norm < 1e-12:
                continue
            C_spherical = C / C_norm
            
            # The angular radius is the angle between C_spherical and any vertex
            dot_val = np.clip(np.dot(C_spherical, v1), -1.0, 1.0)
            radius = np.arccos(dot_val)
            
            # Check that the cap is empty: no other points are closer to C_spherical
            # than the vertices. Actually, for the covering radius, we need the
            # farthest point from all configuration points. The candidate points
            # are the circumcenters of the facets. We need to check that the
            # circumcenter is indeed a local maximum of the distance function.
            
            # For now, just track the maximum radius
            if radius > max_radius:
                max_radius = radius
        
        return np.degrees(max_radius)
        
    except Exception:
        return 180.0

def _gradient_covering_radius(points):
    """
    Compute a subgradient of the covering radius with respect to the points.
    This is used for a projected gradient descent method.
    """
    try:
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        hull = ConvexHull(points)
        
        # Find the facet with the largest circumradius
        max_radius = 0.0
        best_facet = None
        best_C = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            d1 = v2 - v1
            d2 = v3 - v1
            
            A = np.array([[np.dot(d1, d1), np.dot(d1, d2)],
                          [np.dot(d1, d2), np.dot(d2, d2)]])
            b_vec = np.array([np.dot(d1, d1), np.dot(d2, d2)])
            
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) < 1e-15:
                continue
            
            t = np.linalg.solve(A, b_vec)
            C = v1 + t[0]*d1 + t[1]*d2
            
            C_norm = np.linalg.norm(C)
            if C_norm < 1e-12:
                continue
            C_spherical = C / C_norm
            
            dot_val = np.clip(np.dot(C_spherical, v1), -1.0, 1.0)
            radius = np.arccos(dot_val)
            
            if radius > max_radius:
                max_radius = radius
                best_facet = simplex
                best_C = C_spherical
        
        if best_facet is None or best_C is None:
            return np.zeros_like(points)
        
        # The subgradient with respect to the points in the worst facet
        # points them away from the circumcenter
        grad = np.zeros_like(points)
        for idx in best_facet:
            # Gradient of arccos(C·v) w.r.t. v is -C/sqrt(1-(C·v)^2)
            dot_val = np.dot(best_C, points[idx])
            denom = np.sqrt(max(1e-15, 1.0 - dot_val * dot_val))
            grad[idx] = -best_C / denom
        
        return grad
        
    except Exception:
        return np.zeros_like(points)

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1)
    norms = np.maximum(norms, 1e-12)
    return points / norms[:, np.newaxis]

def _lloyd_iteration(points, num_iters=50):
    """
    Perform Lloyd's algorithm iterations for sphere covering.
    Each point moves to the centroid of its Voronoi cell (on the sphere).
    """
    points = _project_to_sphere(points)
    
    for _ in range(num_iters):
        # Compute the Voronoi cells using the convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the facets that contain it
        # and compute the centroid of its Voronoi cell
        new_points = np.zeros_like(points)
        
        # Build a mapping from vertex index to list of adjacent facets
        # hull.vertices gives all vertices
        # hull.simplices gives the facets
        
        # For each point, collect the facet centers (circumcenters) of adjacent facets
        # The Voronoi cell of point i is the set of points on the sphere closer to i
        # than to any other point. Its centroid can be approximated by the average
        # of the circumcenters of the facets adjacent to i.
        
        # Simpler approach: for each point, move it towards the center of mass
        # of the nearest few points' circumcenters.
        
        # Actually, a simpler Lloyd iteration: move each point to the centroid
        # of the points in its Voronoi cell. But computing Voronoi cells on the
        # sphere is complex.
        
        # Alternative: use the fact that the optimal configuration is a critical
        # point of the covering radius. We can use a gradient-based method instead.
        
        break
    
    return points

def construct():
    """
    Construct 50 points on the unit sphere to minimize the covering radius.
    Uses a combination of:
    1. Good initial configurations (Fibonacci spiral, icosahedron-based, random)
    2. Gradient descent on the covering radius
    3. Local optimization with tight tolerance
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 130s out of 150s, leaving 20s margin
    
    best_points = None
    best_radius = np.inf
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib_points)
    
    # 2. Random configurations
    rng = np.random.RandomState(42)
    for _ in range(5):
        random_points = rng.randn(N, 3)
        random_points /= np.linalg.norm(random_points, axis=1)[:, np.newaxis]
        configs.append(random_points)
    
    # 3. Icosahedron-based: start from 12 icosahedron vertices and add 38 more
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1)[:, np.newaxis]
    
    # Add 38 more points near the icosahedron vertices
    ico_based = ico_vertices.copy()
    extra = rng.randn(38, 3)
    extra /= np.linalg.norm(extra, axis=1)[:, np.newaxis]
    ico_based = np.vstack([ico_vertices, extra])
    configs.append(ico_based)
    
    # Evaluate all initial configurations
    for pts in configs:
        if time.time() > deadline:
            break
        radius = _covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()
    
    if best_points is None:
        return fib_points
    
    # Gradient descent on the covering radius
    points = best_points.copy()
    lr = 0.01
    min_lr = 1e-6
    
    for iteration in range(2000):
        if time.time() > deadline - 10.0:
            break
        
        # Compute covering radius
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Compute subgradient
        grad = _gradient_covering_radius(points)
        
        # Project gradient onto the tangent space
        # The gradient should be orthogonal to the radial direction
        points_unit = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        radial = points_unit * np.sum(grad * points_unit, axis=1)[:, np.newaxis]
        tangential_grad = grad - radial
        
        # Update
        points = points - lr * tangential_grad
        points = _project_to_sphere(points)
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr = max(lr * 0.9, min_lr)
    
    # Final local optimization using scipy
    if time.time() < deadline - 20.0:
        def objective(x):
            pts = x.reshape(N, 3)
            pts = _project_to_sphere(pts)
            return _covering_radius(pts)
        
        # Use Nelder-Mead for the final polish
        x0 = best_points.flatten()
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_radius:
                best_radius = result.fun
                best_points = _project_to_sphere(result.x.reshape(N, 3))
        except Exception:
            pass
    
    # Ensure the result is valid
    if best_points is None or not np.all(np.isfinite(best_points)):
        return fib_points
    
    # Final check: ensure no coincident points
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                # Perturb
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points

# EVOLVE-BLOCK-END
