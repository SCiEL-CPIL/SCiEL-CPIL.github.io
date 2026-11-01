"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius(pts):
    """Compute covering radius using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        # For each facet, compute circumcenter and check if it's the worst point
        # The covering radius is the max over all facets of the distance from
        # circumcenter to facet vertices (in angular terms)
        # Actually, for a point set on the unit sphere, the covering radius is
        # determined by the largest empty cap, which corresponds to the facet
        # with the largest circumradius in angular distance.
        
        # For each facet (triangle on the convex hull), compute the circumcenter
        # of the three points and the angular distance to each vertex.
        # The worst facet is the one where the circumcenter (projected to sphere)
        # is farthest from all three vertices.
        
        max_angle = 0.0
        for simplex in hull.simplices:
            v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Compute circumcenter of triangle on unit sphere
            # The circumcenter in 3D is along the normal to the plane
            n = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # The circumcenter of the spherical triangle is at distance
            # such that the angular distance to each vertex is equal.
            # For points on the unit sphere, the circumcenter C satisfies:
            # C . v0 = C . v1 = C . v2 = cos(r) where r is the angular radius
            # C is along the normal n (or -n), and |C| may not be 1.
            # Actually, the circumcenter of a spherical triangle lies on the
            # line through the origin perpendicular to the plane of the triangle.
            
            # Let C = t * n. Then C . v0 = t * (n . v0) = cos(r)
            # We need C . v0 = C . v1 = C . v2, which is satisfied since
            # n is perpendicular to the plane containing v0, v1, v2.
            
            # The angular radius r satisfies cos(r) = t * (n . v0)
            # We want to find r such that the point at angular distance r from
            # each vertex along the sphere is the circumcenter.
            
            # Alternative: compute the circumradius directly.
            # For a spherical triangle with sides a, b, c (angular distances),
            # the circumradius R satisfies a formula, but it's complex.
            
            # Simpler: the covering radius for a facet is the angular distance
            # from the circumcenter to any vertex. The circumcenter on the sphere
            # is the point equidistant from all three vertices.
            
            # For three points on the unit sphere, the circumcenter is:
            # C = (v0 + v1 + v2 + ...) / |...| -- no, that's the centroid.
            
            # Let me use the fact that the circumcenter C satisfies:
            # |C - v0| = |C - v1| = |C - v2| (Euclidean, but C is on the sphere)
            # Actually, for the spherical circumcenter, we want the point on the
            # sphere equidistant (angularly) from v0, v1, v2.
            
            # The angular distance from C to v0 is arccos(C . v0).
            # So we want C . v0 = C . v1 = C . v2.
            # C must be along the normal n (since the set of points equidistant
            # from v0 and v1 is the plane through the origin perpendicular to v0-v1,
            # etc.)
            
            # So C = ±n (normalized). But n is already normalized.
            # Check: n . v0, n . v1, n . v2 should be equal?
            # n is perpendicular to (v1-v0) and (v2-v0), so n . v1 = n . v0 and n . v2 = n . v0.
            # Yes! So n . v0 = n . v1 = n . v2.
            
            cos_r = n . v0
            # The circumcenter could be n or -n. We want the one that gives
            # a valid circumcenter (the cap should be empty, i.e., the circumcenter
            # should be on the "outside" of the triangle).
            
            # The angular radius is arccos(|cos_r|) but we need to be careful.
            # Actually, the circumcenter is the point on the sphere at angular
            # distance r from each vertex, where r = arccos(n . v0) if n . v0 > 0,
            # or r = arccos(-n . v0) if we use -n.
            
            # For the covering radius, we care about the empty cap. The facet
            # defines a cap that is empty of other points. The circumcenter of
            # this cap is the point farthest from all vertices within the cap.
            
            # The angular radius of the circumcircle is:
            r = np.arccos(np.clip(cos_r, -1.0, 1.0))
            # But we need to check which side: the cap is on the side of the
            # facet away from the origin (since the hull is convex and contains
            # the origin for a good covering).
            
            # Actually, for a convex hull containing the origin, each facet's
            # outward normal points away from the origin. The empty cap is on
            # the outside, so the circumcenter is in the direction of the
            # outward normal.
            
            # The outward normal for a facet of the convex hull: scipy's
            # ConvexHull gives facets such that the interior is on the side
            # where the normal points... let's just compute both and take the
            # one that gives a larger radius (worse case).
            
            # The angular distance from the circumcenter to a vertex:
            # If circumcenter is n, distance = arccos(n . v0) = arccos(cos_r)
            # If circumcenter is -n, distance = arccos(-cos_r)
            
            # For the covering radius, we want the maximum over all facets.
            # The relevant circumcenter is the one that lies in the empty cap.
            
            # Let's just compute the angular radius as:
            # The circumradius of the spherical triangle.
            
            # For a triangle with vertices at angular distances a, b, c from
            # the circumcenter, the circumradius R satisfies:
            # sin(a) = sin(R) * ... this is getting complicated.
            
            # Let me just compute the Euclidean circumradius of the triangle
            # and convert, or use a direct formula.
            
            # Actually, the simplest approach: the covering radius is the max
            # over all points on the sphere of the distance to the nearest
            # config point. This is equivalent to the max over all facets of
            # the angular circumradius of the facet's circumcircle.
            
            # For three points on the unit sphere, the angular circumradius R
            # can be computed from the side lengths. Let a, b, c be the
            # angular distances between the vertices.
            
            a = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0))
            b = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            c = np.arccos(np.clip(np.dot(v0, v2), -1.0, 1.0))
            
            # Circumradius of spherical triangle:
            # R = arcsin( (sin(a) * sin(b) * sin(c)) / (2 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c)) )
            # where s = (a+b+c)/2 ... no, that's not right.
            
            # Actually, for a spherical triangle, the circumradius R is given by:
            # tan(R) = (tan(a/2) * tan(b/2) * tan(c/2)) / (sin(s))
            # where s = (a+b+c)/2? I'm not sure of the exact formula.
            
            # Let me use a numerical approach: the circumcenter is along n.
            # The angular distance from n to v0 is arccos(n . v0).
            # This is the circumradius if n is the circumcenter.
            
            # But we need to verify that n is indeed on the sphere and that
            # the cap defined by the facet is empty.
            
            # For now, let's just use arccos(n . v0) as an approximation
            # and refine later.
            
            angle = np.arccos(np.clip(np.abs(cos_r), 0.0, 1.0))
            if angle > max_angle:
                max_angle = angle
        
        return np.degrees(max_angle)
    except Exception:
        return 90.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    def _fibonacci_sphere(n, seed=0):
        rng = np.random.default_rng(seed)
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        # Add small random perturbation for multi-start
        if seed != 0:
            pts = pts + rng.normal(0, 0.01, pts.shape)
            pts = _normalize(pts)
        return pts
    
    # Initial configuration
    pts = _fibonacci_sphere(N, seed=0)
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Refinement loop
    def _refine(pts, max_time=None):
        """Refine points by moving them away from the worst facet."""
        if max_time is not None and time.time() > max_time:
            return pts, _covering_radius(pts)
        
        for iteration in range(200):
            if max_time is not None and time.time() > max_time:
                break
            
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Find the worst facet
            worst_facet = None
            worst_angle = 0.0
            worst_circumcenter = None
            
            for simplex in hull.simplices:
                v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                
                cos_r = n . v0
                # Determine which direction gives the outward normal
                # The outward normal should point away from the centroid of all points
                centroid = np.mean(pts, axis=0)
                if np.dot(n, centroid) > 0:
                    n = -n
                
                angle = np.arccos(np.clip(np.abs(n . v0), 0.0, 1.0))
                if angle > worst_angle:
                    worst_angle = angle
                    worst_facet = simplex
                    worst_circumcenter = n
            
            if worst_facet is None:
                break
            
            # Move the three vertices of the worst facet slightly away from
            # the circumcenter to reduce the circumradius
            idx0, idx1, idx2 = worst_facet
            step_size = 0.01
            
            # Move each vertex in the direction away from the circumcenter
            for idx in [idx0, idx1, idx2]:
                v = pts[idx]
                # Direction away from circumcenter
                dir_away = v - worst_circumcenter
                dir_away = dir_away / np.linalg.norm(dir_away)
                pts[idx] = v + step_size * dir_away
            
            pts = _normalize(pts)
        
        return pts, _covering_radius(pts)
    
    # Multi-start with refinement
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        pts = _fibonacci_sphere(N, seed=s)
        pts, radius = _refine(pts, max_time=deadline - 2.0)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final polish: more iterations on the best
    if time.time() < deadline - 5.0:
        best_pts, best_radius = _refine(best_pts, max_time=deadline - 1.0)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
