"""Covering the sphere with 50 equal caps - optimized via Lloyd's algorithm and gradient descent."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Use convex hull to find facets, then compute circumradius of each facet
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback
    
    max_circ_r = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        pts = points[simplex]  # 3 points for a tetrahedron facet (triangle)
        # Compute circumcenter of the triangle
        a, b, c = pts
        # Vectors
        ab = b - a
        ac = c - a
        # Cross product for normal
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter of a triangle on the unit sphere
        # lies along the normal from the center of the sphere
        # Distance from origin to circumcenter plane
        # For a triangle with vertices a,b,c on unit sphere:
        # circumradius R satisfies: the angular distance from circumcenter to each vertex
        
        # Method: find point o on the normal such that |o - a| = |o - b| = |o - c|
        # o = t * n where t is scalar
        # |t*n - a|^2 = |t*n - b|^2
        # t^2 - 2t(n.a) + |a|^2 = t^2 - 2t(n.b) + |b|^2
        # -2t(n.a) + 1 = -2t(n.b) + 1  (since |a|=|b|=1)
        # t(n.a - n.b) = 0
        # This doesn't work directly. Let me use a different approach.
        
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
        # is the intersection of the perpendicular bisectors.
        
        # Simpler: the covering radius for a facet is the maximum angular distance
        # from any point in the spherical triangle to the nearest of the three vertices.
        # This is equivalent to the circumradius of the spherical triangle.
        
        # For a spherical triangle, the circumradius can be computed as follows:
        # The circumcenter is the point equidistant from all three vertices.
        # On the sphere, this is the point where the great circles that are perpendicular
        # bisectors of the sides meet.
        
        # Alternative: compute the Euclidean circumcenter of the triangle in 3D,
        # then project to the sphere. The angular distance from this point to each vertex
        # is the circumradius.
        
        # Euclidean circumcenter of triangle abc:
        # Solve for o = a + s*ab + t*ac such that |o-a| = |o-b| = |o-c|
        # |o-a|^2 = |o-b|^2 => 2*o.(b-a) = |b|^2 - |a|^2 = 0 (since both on unit sphere)
        # So o.(b-a) = 0 and o.(c-a) = 0
        # o = a + s*ab + t*ac
        # (a + s*ab + t*ac).(b-a) = 0
        # a.(b-a) + s*|ab|^2 + t*ac.(b-a) = 0
        # (a.b - 1) + s*|ab|^2 + t*(ac.b - a.a) = 0  -- wait, ac.(b-a) = ac.b - ac.a
        # Let me just use the standard formula.
        
        # For triangle with vertices on unit sphere:
        # The circumcenter in 3D space (not on sphere) is:
        # o = [a*(|b|^2*|c|^2 - ...) ...] -- this is getting complicated.
        
        # Let me use a numerical approach: minimize max angular distance
        # over the spherical triangle. But that's expensive.
        
        # Actually, the key insight: for the covering radius, we need the maximum
        # over all points on the sphere of the distance to the nearest site.
        # This maximum occurs either at a site (distance 0) or at a point equidistant
        # from 3 or more sites (a Voronoi vertex) or at infinity (not possible on sphere).
        # So the covering radius is the maximum over all Voronoi vertices of the
        # distance to the nearest site.
        
        # For a convex hull facet (triangle), the circumcenter of that triangle
        # (projected to the sphere) is a candidate Voronoi vertex.
        
        # Let me compute the circumcenter properly.
        # Given three points a, b, c on the unit sphere, the circumcenter o
        # (as a point in R^3, then normalized) satisfies:
        # o.a = o.b = o.c (equal dot products, which means equal angular distances)
        # and o is in the plane spanned by a, b, c (which it is if we're looking
        # at the circumcenter of the triangle in 3D).
        
        # Actually, the circumcenter of the spherical triangle is the normalization
        # of the Euclidean circumcenter of the triangle abc.
        
        # Euclidean circumcenter:
        # d = 2*(a*(b x c) + b*(c x a) + c*(a x b)) / |a x b x c| -- no, that's not right.
        
        # Standard formula for circumcenter of triangle with vertices a, b, c:
        # Let d = 2*(a.x*(b.y-c.y) + b.x*(c.y-a.y) + c.x*(a.y-b.y))
        # This is for 2D. For 3D, we need to work in the plane of the triangle.
        
        # Let me use a different approach: solve the linear system.
        # o = alpha*a + beta*b + gamma*c (circumcenter is in the plane of the triangle)
        # Actually no, the circumcenter is not necessarily a convex combination.
        
        # Let me just compute it numerically using the fact that:
        # (o - a).perp = 0 where perp is the perpendicular bisector direction.
        
        # Simpler numerical approach: the circumcenter o satisfies:
        # |o-a|^2 = |o-b|^2 => o.(a-b) = (|a|^2 - |b|^2)/2 = 0
        # |o-a|^2 = |o-c|^2 => o.(a-c) = (|a|^2 - |c|^2)/2 = 0
        # So o is orthogonal to (a-b) and (a-c), meaning o is along the normal n.
        # o = t * n
        # Then |t*n - a|^2 = |t*n - b|^2 is automatically satisfied.
        # We need |t*n - a|^2 = t^2 - 2t*(n.a) + 1
        # The circumradius (Euclidean) is sqrt(t^2 - 2t*(n.a) + 1)
        # But we need the angular distance, which is arccos(n.a) if o = t*n... 
        # Wait, the angular distance from o/|o| to a is arccos((o/|o|).a) = arccos((t*n).a/|t*n|) = arccos(n.a)
        # So the angular circumradius is arccos(n.a) = arccos(n.b) = arccos(n.c)
        
        # But wait, this assumes the circumcenter is along the normal, which is only true
        # if the triangle is "symmetric" in some sense. Let me reconsider.
        
        # Actually, I made an error. The conditions o.(a-b) = 0 and o.(a-c) = 0 mean
        # o is orthogonal to both (a-b) and (a-c), so o is along n = (a-b) x (a-c).
        # This is correct! The circumcenter of a triangle inscribed in the unit sphere
        # lies along the normal to the plane of the triangle.
        
        # So the angular distance from the circumcenter (normalized normal) to any vertex
        # is arccos(n . a). This is the circumradius of the spherical triangle.
        
        dot_a = np.dot(n, a)
        dot_b = np.dot(n, b) 
        dot_c = np.dot(n, c)
        
        # The circumradius is the maximum of these angular distances
        # (they should be equal for a true circumcenter, but numerically they may differ slightly)
        circ_r = np.arccos(np.clip(max(dot_a, dot_b, dot_c), -1.0, 1.0))
        
        # Check if this circumcenter is "inside" the spherical triangle (i.e., the cap is empty)
        # The cap centered at n with radius circ_r should contain a, b, c and no other points.
        # For now, just use the circumradius.
        
        max_circ_r = max(max_circ_r, circ_r)
    
    return np.degrees(max_circ_r)


def _lloyd_polish(points, max_iters=200, tol=1e-10):
    """Lloyd's algorithm: move each point toward the centroid of its Voronoi cell."""
    points = points.copy().astype(np.float64)
    
    for it in range(max_iters):
        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each vertex, find the adjacent facets
        # The Voronoi cell of a vertex is the set of points closer to it than to any other.
        # On the sphere, the Voronoi cell boundaries are great circle arcs.
        # The centroid of the Voronoi cell can be approximated by the average of the
        # circumcenters of the adjacent facets, weighted by the facet area.
        
        # For each vertex, collect adjacent triangles
        vertex_facets = {v: [] for v in range(N)}
        for simplex in hull.simplices:
            for v in simplex:
                vertex_facets[v].append(simplex)
        
        new_points = np.zeros_like(points)
        
        for v in range(N):
            facets = vertex_facets[v]
            if not facets:
                new_points[v] = points[v]
                continue
            
            # For each adjacent facet, compute its circumcenter (normalized normal)
            # and weight by the solid angle (approximated by the area of the spherical triangle)
            centroid = np.zeros(3)
            total_weight = 0.0
            
            for simplex in facets:
                pts = points[simplex]
                a, b, c = pts
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                
                # Ensure the normal points outward (away from origin, i.e., in the direction of the facet)
                # The centroid of the triangle should be in the direction of n
                centroid_tri = (a + b + c) / 3.0
                if np.dot(n, centroid_tri) < 0:
                    n = -n
                
                # Weight by the solid angle of the spherical triangle
                # Approximate solid angle by the Euclidean area of the triangle
                area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
                
                centroid += n * area
                total_weight += area
            
            if total_weight > 1e-12:
                centroid /= total_weight
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    new_points[v] = centroid / centroid_norm
                else:
                    new_points[v] = points[v]
            else:
                new_points[v] = points[v]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        
        if diff < tol:
            break
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Phase 1: Lloyd's algorithm polish
    points_lloyd = _lloyd_polish(points, max_iters=300, tol=1e-12)
    radius_lloyd = _covering_radius(points_lloyd)
    
    if radius_lloyd < best_radius:
        best_points = points_lloyd
        best_radius = radius_lloyd
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_points
    
    # Phase 2: Gradient descent on the covering radius using soft-max approximation
    # We minimize the max angular distance by using a soft-max of the distances
    
    def _objective_and_grad(pts):
        """Compute soft-max covering radius and its gradient."""
        # Reshape to (N, 3)
        pts = pts.reshape(N, 3)
        
        # Compute pairwise dot products
        # dot[i,j] = pts[i].pts[j]
        dots = pts @ pts.T
        np.fill_diagonal(dots, 1.0)
        
        # Angular distance: arccos(dot)
        # For soft-max, we use a temperature parameter
        temp = 0.01  # smaller = sharper max
        
        # For each point on the sphere (approximated by the sites themselves for gradient),
        # the distance to the nearest site is what matters.
        # We use a soft-max over the distances.
        
        # Actually, let's use a different approach: minimize the sum of exp(distance/temp)
        # over a set of test points. But that's expensive.
        
        # Instead, use the fact that the covering radius is determined by the Voronoi vertices.
        # Approximate by considering the circumcenters of the hull facets.
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            return best_radius, np.zeros_like(pts)
        
        # For each facet, compute the circumcenter and its distance to the nearest site
        max_dist = 0.0
        grad = np.zeros_like(pts)
        
        for simplex in hull.simplices:
            a_idx, b_idx, c_idx = simplex
            a, b, c = pts[a_idx], pts[b_idx], pts[c_idx]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # Ensure outward normal
            centroid_tri = (a + b + c) / 3.0
            if np.dot(n, centroid_tri) < 0:
                n = -n
            
            # Angular distance from circumcenter to each vertex
            dist_a = np.arccos(np.clip(np.dot(n, a), -1.0, 1.0))
            dist_b = np.arccos(np.clip(np.dot(n, b), -1.0, 1.0))
            dist_c = np.arccos(np.clip(np.dot(n, c), -1.0, 1.0))
            
            dist = max(dist_a, dist_b, dist_c)
            max_dist = max(max_dist, dist)
            
            # Gradient: move the vertices to reduce the circumradius
            # The circumcenter moves when the vertices move.
            # Approximate gradient: move each vertex slightly toward the circumcenter
            # if it's the farthest, or adjust based on the geometry.
            
            # For now, use a simple heuristic: move each vertex toward the centroid
            # of the circumcenter and the opposite edge midpoint.
            for idx, dist_val in [(a_idx, dist_a), (b_idx, dist_b), (c_idx, dist_c)]:
                if dist_val == dist:
                    # This vertex is the farthest from the circumcenter
                    # Move it toward the circumcenter
                    grad[idx] += 0.1 * (n - pts[idx])
        
        return np.degrees(max_dist), grad
    
    # Use L-BFGS-B with the soft-max objective
    try:
        result = minimize(
            lambda x: _objective_and_grad(x)[0],
            best_points.flatten(),
            jac=lambda x: _objective_and_grad(x)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success and np.all(np.isfinite(result.x)):
            new_points = result.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / np.maximum(norms, 1e-12)
            
            new_radius = _covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_points
    
    # Phase 3: More Lloyd iterations
    points_lloyd2 = _lloyd_polish(best_points, max_iters=500, tol=1e-14)
    radius_lloyd2 = _covering_radius(points_lloyd2)
    
    if radius_lloyd2 < best_radius:
        best_points = points_lloyd2
        best_radius = radius_lloyd2
    
    return best_points
