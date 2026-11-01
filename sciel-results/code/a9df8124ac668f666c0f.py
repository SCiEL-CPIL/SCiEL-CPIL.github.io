"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius in radians for a set of unit vectors.
    
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest configuration point. This equals the
    maximum circumradius of the facets of the convex hull.
    
    For each facet (triangle) of the convex hull, we find the farthest point
    on that facet from the origin (in angular terms). The covering radius is
    the max over all facets of the angular distance from the origin to the
    farthest point on the facet.
    
    Actually, the covering radius is the maximum over all points y on the sphere
    of the distance to the nearest p_i. By duality, this is equivalent to:
    for each facet of the convex hull, the angular radius of the spherical cap
    defined by that facet.
    
    A simpler approach: the covering radius is the maximum, over all points y
    on the sphere, of min_i angle(y, p_i). This can be computed by checking
    the vertices, edges, and face centers of the Voronoi diagram, but a more
    direct approach for the convex hull is:
    
    For each triangular facet with vertices a, b, c (unit vectors), the farthest
    point on the sphere from all p_i that is "covered" by this facet is the point
    on the sphere that maximizes the minimum distance to a, b, c. This is either:
    - The circumcenter of the triangle abc (if it's inside the triangle)
    - One of the edge midpoints (angular)
    - One of the vertices
    
    But actually, for the covering problem, the critical points are the centers
    of the spherical Voronoi cells. Each cell corresponds to a point p_i, and
    its "farthest point" from p_i within the cell determines the local covering
    radius.
    
    A practical approach: compute the convex hull, then for each facet, compute
    the angular distance from the origin to the plane of the facet, which gives
    the angular radius of the cap. The covering radius is the maximum of these.
    
    Wait, let me think more carefully. The convex hull of the points has facets.
    Each facet is a triangle (or polygon) on the convex hull. The spherical cap
    corresponding to a facet is the set of points on the sphere whose nearest
    point among the p_i is... no, that's not right either.
    
    Let me use a different approach. The covering radius is the largest r such
    that there exists a point y on the sphere with all p_i at distance > r from y.
    Equivalently, it's the maximum over y of min_i d(y, p_i).
    
    The maximum of min_i d(y, p_i) occurs at a point that is either:
    1. A vertex of the spherical Voronoi diagram (equidistant to 3+ points)
    2. On an edge of the spherical Voronoi diagram (equidistant to 2 points)
    3. A vertex p_i itself (but that gives distance 0, so not the max)
    
    For the convex hull approach: the facets of the convex hull correspond to
    the "empty" regions. For each facet, the farthest point on the sphere from
    all p_i that is associated with that facet... 
    
    Actually, the cleanest method: compute the convex hull. For each facet,
    find the point on the sphere that is farthest from the origin's projection
    onto the facet's plane, subject to being in the spherical triangle defined
    by the facet. The angular distance from the origin to the facet plane is
    arccos(n · c) where n is the unit normal and c is... 
    
    Let me just use a direct numerical approach: sample many points on the
    sphere and compute the min distance to each p_i, then take the max. But
    this is expensive and may miss the true maximum.
    
    Better: use the fact that the maximum of min_i d(y, p_i) occurs at a point
    equidistant to at least 3 of the p_i (a Voronoi vertex) or equidistant to
    2 (a Voronoi edge endpoint). 
    
    For a robust implementation, I'll compute the convex hull, then for each
    triangular facet, compute the circumcenter of the triangle on the sphere.
    The angular distance from this circumcenter to each of the 3 vertices is
    the local covering radius for that facet. The overall covering radius is
    the maximum over all facets.
    
    For a triangle with vertices a, b, c (unit vectors), the circumcenter on
    the sphere is the point equidistant (in angular distance) to a, b, c. This
    is the normalized vector (a + b + c) only if the triangle is "small" and
    the circumcenter is inside. More generally, we need to solve for the point
    y such that y·a = y·b = y·c = cos(r) and ||y|| = 1.
    
    This is a system of linear equations:
    y·a = cos(r)
    y·b = cos(r)  
    y·c = cos(r)
    ||y|| = 1
    
    From the first three: y is in the null space of [a; b; c] - cos(r)*I... 
    Actually, y·(a-b) = 0 and y·(a-c) = 0, so y is parallel to (a-b)×(a-c).
    So y = ±((a-b)×(a-c)) / ||(a-b)×(a-c)||.
    
    Then cos(r) = y·a, and r = arccos(y·a).
    
    We need to check that this circumcenter is "inside" the spherical triangle,
    i.e., that it's on the same side of each edge as the opposite vertex.
    If not, the maximum on that facet is at one of the edges or vertices.
    
    For simplicity and robustness, let me compute the circumcenter and check
    if it's valid. If the dot products y·a, y·b, y·c are all > 0 and the
    circumcenter is in the right hemisphere, it's likely valid.
    
    Actually, for the covering problem, I think the standard approach is:
    the covering radius is the maximum over all facets of the angular distance
    from the facet's circumcenter to its vertices, where the circumcenter is
    the point on the sphere equidistant to the 3 vertices of the facet.
    """
    points = np.asarray(points)
    n = len(points)
    if n < 3:
        return np.pi  # degenerate case
    
    # Normalize points
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute force with a fine grid
        return _covering_radius_brute(points)
    
    # For each facet, compute the circumcenter and the angular radius
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute circumcenter on the sphere
        # y is parallel to (a-b) x (a-c)
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-10:
            # Degenerate triangle
            r_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            r_ac = np.arccos(np.clip(np.dot(a, c), -1, 1))
            r_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            max_r = max(max_r, max(r_ab, r_ac, r_bc) / 2)
            continue
        
        y = cross / cross_norm
        
        # Check which hemisphere
        # The circumcenter should be on the same side as the origin's projection
        # For the convex hull, the facet's outward normal points away from the origin
        # The circumcenter for the covering should be the one "inside" the hull
        # i.e., the one closer to the origin in angular terms
        
        # Compute angular distances
        d_a = np.arccos(np.clip(np.dot(y, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(y, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(y, c), -1, 1))
        
        # The circumcenter could be y or -y; we want the one that gives
        # a valid (small) radius
        d_a2 = np.arccos(np.clip(np.dot(-y, a), -1, 1))
        d_b2 = np.arccos(np.clip(np.dot(-y, b), -1, 1))
        d_c2 = np.arccos(np.clip(np.dot(-y, c), -1, 1))
        
        # Choose the circumcenter that is equidistant and inside the triangle
        # The correct one should have all three distances equal (up to numerical error)
        # and should be the one "inside" the spherical triangle
        
        # For the covering radius, we want the farthest point from the p_i
        # The circumcenter of the facet gives a local maximum of the min-distance
        # We need the one that is "outside" the convex hull in the angular sense
        
        # Let's use the one that gives a larger angular distance to the vertices
        # (since we're looking for the covering radius, which is the max of min-distances)
        r1 = max(d_a, d_b, d_c)
        r2 = max(d_a2, d_b2, d_c2)
        
        # The correct circumcenter is the one that is on the "far" side from the
        # bulk of the points, i.e., the one with larger angular distances
        # But actually, for a convex hull facet, the outward normal points away
        # from the centroid. The circumcenter for the covering is the one in
        # the direction of the outward normal.
        
        # Compute the outward normal
        centroid = (a + b + c) / 3
        # The outward normal should point away from the origin (since points are
        # on the unit sphere and the hull contains the origin or not)
        
        # For now, take the max of the two candidate radii
        r_facet = max(r1, r2)
        
        # But we need to ensure the circumcenter is actually a valid point
        # (i.e., the spherical triangle contains it). If the triangle is large,
        # the circumcenter might be outside, and the max would be at an edge.
        
        # For robustness, also check edge midpoints
        # Midpoint of arc ab: normalize(a+b)
        mid_ab = a + b
        mid_ab_norm = np.linalg.norm(mid_ab)
        if mid_ab_norm > 1e-10:
            mid_ab = mid_ab / mid_ab_norm
            r_mid_ab = np.arccos(np.clip(np.dot(mid_ab, a), -1, 1))
            r_facet = max(r_facet, r_mid_ab)
        
        mid_bc = b + c
        mid_bc_norm = np.linalg.norm(mid_bc)
        if mid_bc_norm > 1e-10:
            mid_bc = mid_bc / mid_bc_norm
            r_mid_bc = np.arccos(np.clip(np.dot(mid_bc, b), -1, 1))
            r_facet = max(r_facet, r_mid_bc)
        
        mid_ca = c + a
        mid_ca_norm = np.linalg.norm(mid_ca)
        if mid_ca_norm > 1e-10:
            mid_ca = mid_ca / mid_ca_norm
            r_mid_ca = np.arccos(np.clip(np.dot(mid_ca, c), -1, 1))
            r_facet = max(r_facet, r_mid_ca)
        
        max_r = max(max_r, r_facet)
    
    return max_r


def _covering_radius_brute(points, n_samples=10000):
    """Brute force covering radius estimation with random sampling."""
    rng = np.random.RandomState(42)
    # Generate random points on the sphere
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    phi = np.arccos(rng.uniform(-1, 1, n_samples))
    samples = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Compute min distance from each sample to each point
    # dist^2 = 2 - 2 * dot product
    dots = samples @ points.T
    min_dots = np.max(dots, axis=1)  # max dot = min distance
    max_angle = np.arccos(np.clip(min_dots, -1, 1))
    return np.max(max_angle)


def _objective_and_grad(x, n_points=N):
    """Objective: covering radius, with gradient for optimization.
    
    x: flattened array of 3*N values
    Returns: (covering_radius, gradient)
    """
    points = x.reshape(n_points, 3)
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    p = points / norms
    
    # Compute covering radius
    r = _covering_radius(p)
    
    # For gradient, we use a simple numerical approach or skip
    # Since we're using L-BFGS-B, we need a gradient
    # For now, return zero gradient and let L-BFGS-B use numerical gradients
    grad = np.zeros_like(x)
    
    return r, grad


def _objective(x, n_points=N):
    """Objective function for optimization."""
    points = x.reshape(n_points, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    p = points / norms
    return _covering_radius(p)


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_r = _covering_radius(best_points)
    
    # Optimization using L-BFGS-B
    x0 = points.flatten()
    
    # Use a smaller number of iterations to stay within time budget
    # L-BFGS-B with numerical gradients
    result = None
    try:
        # Use a callback to check time
        def callback(xk):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                raise TimeoutError()
            return False
        
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8},
            callback=callback
        )
    except (TimeoutError, Exception) as e:
        pass
    
    if result is not None and result.success:
        new_points = result.x.reshape(N, 3)
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        new_points = new_points / norms
        
        new_r = _covering_radius(new_points)
        if new_r < best_r:
            best_points = new_points
            best_r = new_r
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try a few more local optimizations from perturbed starts
        rng = np.random.RandomState(123)
        for trial in range(5):
            if time.time() - start_time > deadline - 10:
                break
            
            # Perturb the best points
            noise = rng.normal(0, 0.1, size=(N, 3))
            perturbed = best_points + noise
            x0 = perturbed.flatten()
            
            try:
                result2 = minimize(
                    _objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8},
                    callback=callback
                )
                
                if result2.success:
                    new_points = result2.x.reshape(N, 3)
                    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    new_points = new_points / norms
                    
                    new_r = _covering_radius(new_points)
                    if new_r < best_r:
                        best_points = new_points
                        best_r = new_r
            except (TimeoutError, Exception):
                pass
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
