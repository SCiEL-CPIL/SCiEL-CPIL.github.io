"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius_degrees(pts):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest configuration point.
    
    This equals the maximum circumradius of the facets of the convex hull
    of the points (each facet's circumcenter projected to the sphere gives
    the farthest point from that facet's vertices).
    """
    if len(pts) < 3:
        return 180.0
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_r = 0.0
    for simplex in hull.simplices:
        # Get the facet vertices
        facet_pts = pts[simplex]
        n = len(facet_pts)
        
        # The circumcenter of the facet in 3D: for a simplex of n+1 points in R^3
        # (so n=2 for triangles, which is typical for convex hull in 3D)
        # We need the center of the sphere passing through all facet vertices
        # that lies on the normal direction from the facet.
        
        # For a triangle (n=2, 3 vertices):
        # The circumcenter of the triangle in 3D, projected onto the normal
        # Actually, the covering radius for a facet is the angular distance from
        # the facet's circumcenter (on the sphere) to any vertex.
        
        # The key insight: for a set of points on the unit sphere, the farthest
        # point from the nearest point is at the circumcenter of some facet of
        # the convex hull. The angular distance from a vertex to this circumcenter
        # is the covering radius contribution.
        
        # Compute the circumcenter of the facet (the point equidistant from all
        # facet vertices, lying on the line through the centroid in the normal direction)
        
        # For 3 points (triangle):
        if n == 2:
            # Three points a, b, c on unit sphere
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            
            # Normal to the facet
            nrm = np.cross(b - a, c - a)
            nrm_norm = np.linalg.norm(nrm)
            if nrm_norm < 1e-12:
                continue
            nrm = nrm / nrm_norm
            
            # Circumcenter of triangle in 3D:
            # Solve for point p = a + s*(b-a) + t*(c-a) such that
            # |p-a|^2 = |p-b|^2 = |p-c|^2
            # This gives the in-plane circumcenter. Then project to sphere along normal.
            
            # Actually, the circumcenter on the sphere is the point on the normal
            # line (through the in-plane circumcenter) that lies on the unit sphere.
            
            # In-plane circumcenter:
            d = 2.0 * np.dot(a, np.cross(b, c))
            if abs(d) < 1e-12:
                continue
            
            # Circumcenter formula for triangle on unit sphere
            # C = (|b|^2|c|^2(a-b) + |c|^2|a|^2(b-c) + |a|^2|b|^2(c-a)) / (2 * det[a,b,c])
            # Since all on unit sphere:
            # C = (|c|^2(a-b) + |a|^2(b-c) + |b|^2(c-a)) / (2 * det)
            # Wait, let me use the standard formula.
            
            # For points on unit sphere, the circumcenter of the triangle (as a point
            # in 3D space, not on the sphere) is:
            # C = [ |b|^2|c|^2 (a) + |c|^2|a|^2 (b) + |a|^2|b|^2 (c) ] / (2 * |a . (b x c)|)
            # Since |a|=|b|=|c|=1:
            # C = (a + b + c) / (2 * |a . (b x c)|)  ... NO, that's not right.
            
            # Let me just compute it directly.
            # The circumcenter P satisfies |P-a|^2 = |P-b|^2 = |P-c|^2
            # and P lies in the plane of a,b,c.
            
            # Set up linear system:
            # 2(b-a).P = |b|^2 - |a|^2 = 0 (since both on unit sphere)
            # 2(c-a).P = |c|^2 - |a|^2 = 0
            # P = a + alpha*(b-a) + beta*(c-a)
            
            # Actually simpler: since all on unit sphere, the circumcenter of the
            # triangle in 3D is:
            # P = (|b|^2 (c-a) x (b-a) + ...) hmm let me just solve the linear system.
            
            # 2*(b-a).P = 0
            # 2*(c-a).P = 0
            # And P = a + s*(b-a) + t*(c-a)
            
            # From first eq: 2*(b-a).(a + s*(b-a) + t*(c-a)) = 0
            # => 2*(b-a).a + 2*s*|b-a|^2 + 2*t*(b-a).(c-a) = 0
            # From second eq: 2*(c-a).a + 2*s*(c-a).(b-a) + 2*t*|c-a|^2 = 0
            
            ba = b - a
            ca = c - a
            A = np.array([
                [np.dot(ba, ba), np.dot(ba, ca)],
                [np.dot(ca, ba), np.dot(ca, ca)]
            ])
            rhs = np.array([
                -np.dot(ba, a),
                -np.dot(ca, a)
            ])
            
            try:
                st = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            
            P = a + st[0] * ba + st[1] * ca
            
            # P is the circumcenter in the plane. Now project to unit sphere along normal.
            # The point on the unit sphere in the direction of P (from origin) is P/|P|
            P_norm = np.linalg.norm(P)
            if P_norm < 1e-12:
                continue
            P_sphere = P / P_norm
            
            # Angular distance from P_sphere to a (or b or c, all same)
            cos_angle = np.clip(np.dot(P_sphere, a), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            max_r = max(max_r, angle)
            
        elif n == 1:
            # Two points (edge) - shouldn't happen for 3D convex hull typically
            a, b = facet_pts[0], facet_pts[1]
            # Midpoint direction
            mid = a + b
            mid_norm = np.linalg.norm(mid)
            if mid_norm < 1e-12:
                continue
            mid_sphere = mid / mid_norm
            cos_angle = np.clip(np.dot(mid_sphere, a), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            max_r = max(max_r, angle)
            
        elif n == 3:
            # Four points (tetrahedron face) - unlikely for convex hull in 3D
            # Just use the centroid as approximation
            centroid = np.mean(facet_pts, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid_sphere = centroid / centroid_norm
            for fp in facet_pts:
                cos_angle = np.clip(np.dot(centroid_sphere, fp), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                max_r = max(max_r, angle)
    
    return np.degrees(max_r)


def _worst_facet_circumcenters(pts):
    """
    For each point, find the worst adjacent facet and return the direction
    to move (toward the circumcenter of the worst facet).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    
    # Map each point to its adjacent facets
    point_to_facets = {i: [] for i in range(len(pts))}
    for idx, simplex in enumerate(hull.simplices):
        for p in simplex:
            point_to_facets[p].append(idx)
    
    # Compute circumcenter for each facet
    facet_centers = {}
    for idx, simplex in enumerate(hull.simplices):
        facet_pts = pts[simplex]
        n = len(facet_pts)
        
        if n == 2:
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            ba = b - a
            ca = c - a
            A = np.array([
                [np.dot(ba, ba), np.dot(ba, ca)],
                [np.dot(ca, ba), np.dot(ca, ca)]
            ])
            rhs = np.array([
                -np.dot(ba, a),
                -np.dot(ca, a)
            ])
            try:
                st = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            P = a + st[0] * ba + st[1] * ca
            P_norm = np.linalg.norm(P)
            if P_norm < 1e-12:
                continue
            facet_centers[idx] = P / P_norm
        elif n >= 1:
            centroid = np.mean(facet_pts, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                facet_centers[idx] = centroid / centroid_norm
    
    # For each point, find the worst facet
    directions = np.zeros_like(pts)
    for i in range(len(pts)):
        if i not in point_to_facets or not point_to_facets[i]:
            continue
        worst_angle = 0.0
        worst_dir = None
        for fidx in point_to_facets[i]:
            if fidx not in facet_centers:
                continue
            fc = facet_centers[fidx]
            cos_a = np.clip(np.dot(fc, pts[i]), -1.0, 1.0)
            angle = np.arccos(cos_a)
            if angle > worst_angle:
                worst_angle = angle
                worst_dir = fc
        if worst_dir is not None:
            directions[i] = worst_dir - pts[i]
    
    return directions, hull


def _optimize_from_start(pts, max_iters=500, step_size=0.01):
    """
    Local optimization: iteratively move each point toward the circumcenter
    of its worst adjacent facet.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_radius = _covering_radius_degrees(pts)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        directions, _ = _worst_facet_circumcenters(pts)
        if directions is None:
            break
        
        # Check if all directions are zero (converged)
        if np.all(np.linalg.norm(directions, axis=1) < 1e-8):
            break
        
        # Move points
        new_pts = pts + step_size * directions
        new_pts = _normalize(new_pts)
        
        # Check for coincident points
        min_dist = np.inf
        for i in range(len(new_pts)):
            for j in range(i+1, len(new_pts)):
                d = np.linalg.norm(new_pts[i] - new_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-6:
            step_size *= 0.5
            continue
        
        new_radius = _covering_radius_degrees(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Try smaller step
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_pts, best_radius


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturb(pts, sigma=0.1, seed=None):
    """Add Gaussian perturbation and renormalize."""
    rng = np.random.RandomState(seed)
    perturbed = pts + sigma * rng.randn(*pts.shape)
    return _normalize(perturbed)


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_pts = None
    best_radius = float('inf')
    
    # Multiple random starts with local optimization
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        if start_idx == 0:
            # Start from Fibonacci spiral
            pts = _fibonacci_sphere(N)
        elif start_idx == 1:
            # Slightly perturbed Fibonacci
            pts = _perturb(_fibonacci_sphere(N), sigma=0.05, seed=42)
        else:
            # Random starts
            seed = start_idx * 1000 + 7
            pts = _random_sphere(N, seed=seed)
        
        # Run local optimization
        opt_pts, opt_radius = _optimize_from_start(pts, max_iters=300, step_size=0.05)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts
    
    # If we have time, do a second round of refinement with smaller steps
    if best_pts is not None and time.time() - _START_TIME < _DEADLINE - 15:
        # Refine the best solution
        refined_pts, refined_radius = _optimize_from_start(best_pts, max_iters=200, step_size=0.01)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_pts = refined_pts
    
    # Final check: make sure points are valid
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure no coincident points and not in a hemisphere
    best_pts = _normalize(best_pts)
    
    # Check hemisphere: all points should not be in a closed hemisphere
    # A simple check: the convex hull should contain the origin
    try:
        hull = ConvexHull(best_pts)
        # Check if origin is inside the hull
        # For a convex polytope, origin is inside if all facet normals point outward
        # and the origin is on the same side as the interior
        # Simple check: for each facet, the origin should be on the opposite side from the outward normal
        # Actually, let's just check that the points span all of R^3
        # and that the convex hull is full-dimensional
        if hull.volume < 1e-6:
            # Degenerate, try Fibonacci
            best_pts = _fibonacci_sphere(N)
    except Exception:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
