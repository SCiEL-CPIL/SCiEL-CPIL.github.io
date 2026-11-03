"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _project_to_sphere(x):
    return _normalize(x)

def _get_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    The circumradius is the angular distance from the circumcenter to any vertex.
    The circumcenter is the normalized cross product of the edge vectors? No, it's the point on the sphere equidistant from p1, p2, p3.
    It is given by the normalized sum of p1, p2, p3 if they are close, but generally it's the normalized vector perpendicular to the plane of the triangle?
    Actually, the circumcenter of a spherical triangle is the intersection of the great circles perpendicular to the sides.
    A simpler way: The circumradius R satisfies cos(R) = (p1 . p2 + p2 . p3 + p3 . p1 - 1) / 2? No.
    
    Let's use the property that the circumcenter c is proportional to p1 + p2 + p3 only if the triangle is acute?
    Actually, the circumcenter c is the normalized vector:
    c = (p1 + p2 + p3) / ||p1 + p2 + p3|| is NOT generally correct.
    
    The circumcenter is the point c on the unit sphere such that c.p1 = c.p2 = c.p3 = cos(R).
    This implies c is in the span of p1, p2, p3.
    c = a*p1 + b*p2 + c*p3.
    c.p1 = c.p2 = c.p3.
    
    Alternatively, the circumradius can be calculated from the area and side lengths, or simply by finding the point on the sphere that maximizes the minimum distance to the vertices? No, that's the incenter.
    The circumcenter is the point that minimizes the variance? No.
    
    Let's use the formula for the circumradius of a spherical triangle.
    Let a, b, c be the side lengths (angular distances).
    Let S be the spherical excess (area).
    tan(R) = tan(a/2) * tan(b/2) * tan(c/2) / (1 + 2 cos(a/2) cos(b/2) cos(c/2) - cos(a/2) cos(b/2) - cos(b/2) cos(c/2) - cos(c/2) cos(a/2))?
    This is getting complicated.
    
    Simpler approach: The circumcenter is the normalized vector:
    c = (p1 x p2) x (p2 x p3) ... no.
    
    Let's just compute the circumcenter explicitly.
    The circumcenter c satisfies:
    c . p1 = c . p2 = c . p3
    ||c|| = 1
    
    This is a linear system:
    (p1 - p2) . c = 0
    (p2 - p3) . c = 0
    c . c = 1
    
    The solution is the normalized cross product of the two normal vectors to the planes defined by the points?
    No, the vector c is orthogonal to (p1-p2) and (p2-p3).
    So c is parallel to (p1-p2) x (p2-p3).
    Let n = (p1 - p2) x (p2 - p3).
    Then c = _normalize(n).
    
    Let's verify.
    If c is parallel to n, then c . (p1-p2) = 0 and c . (p2-p3) = 0.
    So c.p1 = c.p2 and c.p2 = c.p3.
    Thus c.p1 = c.p2 = c.p3 = cos(R).
    So R = arccos(c.p1).
    
    We must ensure c is on the correct side (the one containing the triangle interior).
    The vector n = (p1-p2) x (p2-p3) points towards the center of the sphere if the triangle is oriented one way, and away if the other.
    We want the circumcenter to be the one "inside" the cap.
    Actually, the circumradius is the distance from the circumcenter to the vertices.
    If we pick the wrong normal, we get the antipodal circumcenter, which has the same angular distance to the vertices?
    cos(R) = c.p1. If we use -c, cos(R') = -c.p1 = -cos(R) = cos(pi-R).
    So R' = pi - R.
    The covering radius is the maximum distance to the nearest point.
    For a facet, the "worst" point is the circumcenter if it lies within the facet.
    If the circumcenter is the antipodal one, it's not the one inside the cap defined by the facet.
    We need the circumcenter that is close to the vertices.
    So we should check if c.p1 > 0. If not, flip c.
    
    """
    n = np.cross(p1 - p2, p2 - p3)
    c = _normalize(n)
    # Ensure c is on the same side as the vertices (dot product positive)
    if np.dot(c, p1) < 0:
        c = -c
    cos_r = np.dot(c, p1)
    # Clamp for numerical stability
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r, c

def _compute_covering_radius(points):
    """
    Compute the covering radius of the configuration.
    This is the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_facet = None
        worst_center = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            r, c = _get_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet = simplex
                worst_center = c
        return max_r, worst_facet, worst_center
    except Exception:
        # If convex hull fails, return a large value
        return np.pi, None, None

def _lloyd_relaxation(points, max_iters=50, tol=1e-6):
    """
    Perform Lloyd's relaxation to maximize the minimum distance between points.
    This helps in spreading points out evenly, which is a good prerequisite for covering.
    """
    pts = points.copy()
    for _ in range(max_iters):
        # Assign each point to the nearest neighbor? No, Lloyd's for covering is different.
        # For covering, we want to minimize the maximum distance to the nearest point.
        # This is equivalent to maximizing the minimum distance from any point on the sphere to the set.
        # A common heuristic is to move each point to the centroid of its Voronoi cell (on the sphere).
        # However, computing spherical Voronoi cells is expensive.
        
        # Alternative: Use a gradient descent approach on the covering radius.
        # But for now, let's stick to the geometric facet shrinking method directly, 
        # initialized with a good start.
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    # Iterative Geometric Relaxation
    # Identify the worst facet and move its vertices away from the circumcenter.
    
    for it in range(1000):
        if time.time() > deadline:
            break
            
        current_radius, worst_facet, worst_center = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if worst_facet is None:
            break
            
        # Move the vertices of the worst facet
        # The gradient of the circumradius with respect to the vertices points towards the circumcenter.
        # So we move the vertices away from the circumcenter.
        
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        # Step size: proportional to the radius?
        # Let's try a small step.
        step = 0.05
        
        # Move v0 away from c
        # v0_new = v0 - step * (c - v0) ? No, c is on the sphere.
        # The direction to move is the projection of (c - v) onto the tangent plane at v.
        # grad = c - (c.v) v
        # We want to decrease the distance to c? No, we want to increase the distance from c to the vertices?
        # Wait, the circumradius is the distance from c to the vertices.
        # If we move the vertices away from c, the triangle expands?
        # No, the circumcenter is determined by the vertices.
        # If we move the vertices away from the current circumcenter, the new circumcenter will move, and the radius might decrease.
        # This is a known heuristic for shrinking the largest facet.
        
        def move_vertex(v, c, step):
            # Projection of c onto tangent space at v
            # grad = c - (c.dot(v)) * v
            # We want to move v in the direction of -grad? 
            # If we move v towards c, the distance v-c decreases?
            # Let's think about a 2D circle. Center c, points on circle.
            # If we move a point on the circle towards the center, it leaves the circle.
            # We are constrained to the sphere.
            # Moving v towards c (along the geodesic) will make the triangle smaller?
            # If we shrink the triangle, its circumradius decreases.
            # So we should move the vertices TOWARDS the circumcenter.
            
            # Geodesic step towards c
            # v_new = _normalize(v + step * (c - (c.dot(v))*v))
            # Wait, if we move v towards c, v and c get closer.
            # The circumradius is the distance from the NEW circumcenter to the NEW vertices.
            # This is a coupled system.
            # Empirically, moving the vertices of the largest facet towards their circumcenter shrinks the facet.
            
            grad = c - np.dot(c, v) * v
            grad = _normalize(grad)
            v_new = v + step * grad
            return _normalize(v_new)
        
        # Update vertices
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        # Optional: Add a small random perturbation to escape local minima?
        # The task says "deterministic, non-stochastic". So we stick to deterministic.
        
    # Final polish with high precision
    # Use scipy.optimize.minimize on the covering radius?
    # The covering radius is non-smooth (max of smooth functions).
    # We can use a smooth approximation or just run more iterations of the geometric method with smaller steps.
    
    # Let's do a final refinement loop with smaller steps
    step = 0.01
    for it in range(500):
        if time.time() > deadline:
            break
        current_radius, worst_facet, worst_center = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        if worst_facet is None:
            break
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            grad = _normalize(grad)
            v_new = v + step * grad
            return _normalize(v_new)
            
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)

    # Check if best_points is valid
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return points
# EVOLVE-BLOCK-END
