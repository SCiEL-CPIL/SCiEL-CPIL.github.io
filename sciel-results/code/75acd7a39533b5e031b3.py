"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
MAX_TIME = 150.0  # evaluator kills after 150s; keep safety margin

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum over all facets of the convex hull
    of the circumradius of that facet (the angular distance from the facet
    center to the farthest vertex, measured as the angle at the sphere center).
    
    For a facet with vertices v1, ..., vk on the unit sphere, the circumradius
    (angular distance) is the angle between the facet centroid direction and
    the farthest vertex. More precisely, for a facet, the covering radius
    contribution is the maximum angular distance from the facet's circumcenter
    (on the sphere) to any vertex of the facet.
    
    Actually, the correct interpretation: for each facet of the convex hull,
    consider the spherical cap centered at the outward normal of the facet
    (normalized) that just touches the facet vertices. The angular radius of
    this cap is the covering radius contribution for that facet.
    
    For a planar facet with vertices on the unit sphere, the circumcenter of
    the facet (in 3D, the center of the circumscribed circle of the facet)
    projected to the sphere gives the point farthest from the facet plane.
    The angular distance from this point to any facet vertex is the same for
    all vertices (if the facet is cyclic, which it is for a planar polygon on
    a sphere... actually not necessarily).
    
    Let me think more carefully. The covering radius is the maximum over all
    points y on the sphere of the angular distance to the nearest chosen point.
    This maximum is achieved at a point that is equidistant from 3 or more
    chosen points (a vertex of the Voronoi diagram on the sphere), which
    corresponds to a facet of the convex hull.
    
    For a facet of the convex hull with vertices v1, ..., vk, the point on the
    sphere farthest from all of them (in the direction away from the origin)
    is along the normal to the facet plane. The angular distance from this
    normal direction to each vertex vi is the same if and only if the facet
    is regular. In general, the covering radius for that facet is the maximum
    angular distance from the facet's circumcenter (on the sphere) to the
    vertices.
    
    A simpler approach: for each facet, compute the circumradius of the facet
    as a spherical polygon. The angular distance from the centroid of the
    facet (normalized) to each vertex gives an upper bound. But the exact
    covering radius for a facet is determined by the circumscribed circle.
    
    Let's use a robust method: for each facet, find the point on the sphere
    that maximizes the minimum angular distance to the facet's vertices.
    This is equivalent to finding the center of the largest empty circle
    with respect to the facet vertices.
    
    Actually, the standard approach: the covering radius is the maximum over
    all facets of the circumradius of the facet. For a facet with vertices
    v1, ..., vk on the unit sphere, the circumradius (angular) can be computed
    as follows: find the circumcenter of the planar polygon (in 3D space),
    then the angular distance from the origin to this circumcenter projected
    onto the sphere... 
    
    Let me use a different, more direct approach. For each facet, the relevant
    quantity is the maximum angular distance from any point in the facet's
    "cell" to the nearest vertex. The worst point is the circumcenter of the
    facet (if it lies within the facet) or a vertex of the facet.
    
    For a convex polygon on the sphere, the point farthest from all vertices
    (in terms of angular distance) within the polygon is either the circumcenter
    (if it's inside) or on the boundary.
    
    Let's compute it as: for each facet, compute the circumradius of the
    planar polygon formed by the vertices. The angular circumradius is
    arccos(1 - R^2 / (2 * d^2)) where R is the planar circumradius and d is
    the distance from the origin to the facet plane. Actually, this is getting
    complicated.
    
    Simpler: for each facet with vertices v1, ..., vk, the covering radius
    contribution is the maximum over i of the angular distance from the
    facet normal (normalized) to vi. But this is only correct if the facet
    normal direction is equidistant from all vertices, which it's not in general.
    
    Let me just compute the actual maximum: for each facet, the worst point
    is the one that maximizes the minimum distance to the vertices. For a
    small number of vertices per facet (typically 3-5), we can find this by
    checking the circumcenter and the vertices/edges.
    
    For practical purposes, a good approximation that's exact for triangles:
    For a triangular facet with vertices a, b, c, the circumcenter on the
    sphere is the normalized cross product or the point equidistant from all
    three. The angular distance from this point to each vertex is the same
    and equals the circumradius.
    
    For a triangle with vertices a, b, c on the unit sphere, the circumradius
    (angular) can be computed as follows:
    - The circumcenter in 3D is the intersection of the perpendicular bisectors.
    - Alternatively, for a triangle inscribed in a circle of radius R (planar)
      at distance d from the origin, the angular radius is arccos(d/R) ... no.
    
    Let me use: for triangle with sides (angular) a, b, c, the circumradius
    R satisfies: sin(R) = (a*b*c) / (4 * area) where area is the spherical
    area... this is getting too complex.
    
    Direct approach: for each facet, compute the circumcenter by solving for
    the point x on the unit sphere equidistant from all facet vertices.
    For a triangle, this is: x = normalize(a + b + c) is NOT correct in general.
    
    Correct method for triangle circumcenter on sphere:
    The circumcenter is the normalized vector in the direction of the normal
    to the plane of the triangle, but offset. Actually, for a triangle on the
    unit sphere, the circumcenter is the point where the perpendicular
    bisectors of the sides meet.
    
    Let me just use: for each facet, compute the maximum angular distance
    from the centroid (normalized) to each vertex. This is an upper bound
    on the true covering radius for that facet, but it's a reasonable
    approximation. For better accuracy, I'll compute the circumradius properly.
    
    For a planar polygon with vertices p1, ..., pk at distance d from origin,
    with planar circumradius R, the angular circumradius is arccos(d/R) if
    the circumcenter is on the far side, or arccos(-d/R) ... 
    
    OK let me just do it numerically for small facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        # For a simplex (triangle in 3D), compute the circumradius
        # The circumcenter is the point equidistant from all vertices
        # For a triangle with vertices a, b, c:
        # circumcenter = normalize(a + b + c) is wrong
        # Correct: solve for x such that |x-a|^2 = |x-b|^2 = |x-c|^2, |x|=1
        
        if len(verts) == 3:
            a, b, c = verts
            # Circumcenter of triangle on unit sphere
            # x = alpha*a + beta*b + gamma*c with constraints
            # |x|^2 = 1, x.a = x.b = x.c
            # This means x is in the direction of the normal to the plane
            # through a,b,c, but at the right distance.
            
            # The plane through a,b,c has normal n = (b-a) x (c-a)
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                # Degenerate
                max_circumradius = max(max_circumradius, np.max(np.arccos(np.clip(-1, 1, np.sum(a * b, keepdims=True)))))
                continue
            n = n / n_norm
            
            # Distance from origin to plane
            d = np.dot(a, n)
            # Planar circumradius
            # For triangle with sides, planar circumradius R_planar
            # Side lengths
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            # Area of triangle
            area = 0.5 * n_norm
            if area < 1e-12:
                max_circumradius = max(max_circumradius, 180.0)
                continue
            R_planar = (ab * bc * ca) / (4 * area)
            
            # The circumcenter is at distance d from origin along n
            # The angular circumradius is the angle from the circumcenter
            # (on the sphere) to a vertex
            # Circumcenter on sphere: n * (1/d) ... no
            # The circumcenter in 3D is at n * d (on the normal line through origin)
            # Wait, the circumcenter of the triangle lies on the line through
            # the origin in direction n (if the triangle is centered at origin)
            # Actually, the circumcenter of a triangle inscribed in a circle
            # in a plane at distance d from origin, with the circle centered
            # on the normal through the origin, is at position n * d.
            # The angular distance from this point (normalized) to a vertex a is:
            # cos(angle) = dot(n, a) = d
            # So angle = arccos(d)
            # But wait, this is the angle from the normal to the vertex,
            # which is the same for all vertices. So the circumradius is arccos(d).
            
            # But we need to check: is the circumcenter on the correct side?
            # The circumcenter should be on the side away from the origin
            # for the covering radius (the far side).
            # Actually, for the convex hull, the facet normal points outward.
            # The covering radius is the angular distance from the outward
            # circumcenter to the vertices.
            
            # cos(angle) = |d| if we take the absolute value
            # But we want the angular distance, which is arccos(|d|) if d != 0
            
            circumradius = np.arccos(np.clip(abs(d), -1, 1))
            max_circumradius = max(max_circumradius, circumradius)
        else:
            # For higher-dimensional simplices (shouldn't happen in 3D)
            # Just use the maximum distance from centroid
            centroid = np.mean(verts, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            for v in verts:
                angle = np.arccos(np.clip(np.dot(centroid, v), -1, 1))
                max_circumradius = max(max_circumradius, angle)
    
    return np.degrees(max_circumradius)


def _gradient_covering_radius(pts):
    """
    Compute a gradient for the covering radius objective.
    This is approximate: we identify the facet(s) that achieve the maximum
    circumradius and compute gradients with respect to the vertices of those facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.zeros_like(pts)
    
    grad = np.zeros_like(pts)
    max_circumradius = 0.0
    worst_facet = None
    
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) == 3:
            a, b, c = verts
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            d = np.dot(a, n)
            circumradius = np.arccos(np.clip(abs(d), -1, 1))
            if circumradius > max_circumradius:
                max_circumradius = circumradius
                worst_facet = (simplex, n, d, a, b, c)
    
    if worst_facet is None:
        return grad
    
    simplex, n, d, a, b, c = worst_facet
    idx_a, idx_b, idx_c = simplex
    
    # Gradient of circumradius = arccos(|d|) with respect to each vertex
    # d = dot(a, n), n = normalize(cross(b-a, c-a))
    # This is complex; use finite differences for the gradient
    
    eps = 1e-7
    grad_fd = np.zeros_like(pts)
    for i in range(3):
        vertex = [a, b, c][i]
        idx = [idx_a, idx_b, idx_c][i]
        for j in range(3):
            perturbation = np.zeros(3)
            perturbation[j] = eps
            pts_pert = pts.copy()
            pts_pert[idx] = _normalize(pts_pert[idx] + perturbation)
            cr_pert = _covering_radius(pts_pert)
            grad_fd[idx, j] = (cr_pert - np.degrees(max_circumradius)) / eps
    
    return grad_fd


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize(pts, max_time=100.0):
    """
    Optimize the covering radius using gradient descent with line search.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    start_time = time.time()
    
    lr = 0.01
    patience = 50
    no_improve = 0
    
    while time.time() - start_time < max_time:
        # Compute gradient
        grad = _gradient_covering_radius(pts)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Gradient descent step
        new_pts = _normalize(pts - lr * grad)
        
        # Evaluate
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            # Reduce learning rate slowly
            lr *= 0.99
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                # Restart with smaller learning rate
                lr = 0.001
                no_improve = 0
                # Try a random perturbation to escape local minimum
                pts = _normalize(pts + 0.01 * np.random.randn(N, 3))
        
        # Check time
        if time.time() - start_time > max_time - 5:
            break
    
    return best_pts, best_cr


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    
    # Run optimization
    optimized_pts, best_cr = _optimize(pts, max_time=MAX_TIME - 20)
    
    # Safety: ensure we return before the deadline
    if time.time() - start_time > MAX_TIME - 5:
        # Return best found so far
        return _normalize(optimized_pts)
    
    # Final normalization
    result = _normalize(optimized_pts)
    
    # Verify validity
    if not np.all(np.isfinite(result)):
        return _normalize(_fibonacci_sphere(N))
    
    return result
# EVOLVE-BLOCK-END
