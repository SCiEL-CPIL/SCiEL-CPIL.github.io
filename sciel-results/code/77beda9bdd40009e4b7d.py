import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facet circumradii."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        return 90.0
    pts = pts / norms
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    # For each triangular facet, compute the circumradius (angular)
    # The circumcenter of a spherical triangle is the point equidistant from all three vertices
    # The covering radius is the max over facets of the angular distance from circumcenter to vertices
    max_circumradius = 0.0
    simplices = hull.simplices
    for tri in simplices:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Compute circumcenter on sphere: solve for point p such that p·a = p·b = p·c
        # This is the intersection of the sphere with the plane equidistant from a,b,c
        # Use the fact that circumcenter lies in the plane spanned by a,b,c
        # Circumcenter = (a+b+c) / |a+b+c| for equilateral, but generally:
        # Solve: p·a = p·b, p·a = p·c, |p|=1
        # p = alpha*a + beta*b + gamma*c
        # Simpler: the circumcenter is along the direction of the cross product of (b-a) and (c-a) 
        # but that gives the normal. Actually, for a spherical triangle, the circumcenter is:
        # p = (a + b + c) normalized doesn't work in general.
        # Better: solve the linear system for p in the plane of a,b,c
        # p·(a-b) = 0, p·(a-c) = 0, |p| = 1
        # The vector a-b and a-c define the plane. p is perpendicular to both in 3D? No.
        # p lies in the plane spanned by a,b,c (which is a 2D subspace through origin)
        # So p = x*a + y*b + z*c, and p·a = p·b = p·c
        # This gives a linear system. Let's use a different approach:
        # The circumcenter is the point on the sphere that is equidistant from a, b, c.
        # It lies on the intersection of the sphere and the two great circles that are perpendicular bisectors.
        # Numerically: minimize |p·a - p·b| + |p·a - p·c| over unit p
        # Or: the circumcenter is in the direction of the vector that is the solution to:
        # (a-b)·p = 0, (a-c)·p = 0 => p is perpendicular to both (a-b) and (a-c)
        # So p = (a-b) × (a-c) normalized? That's the normal to the plane, not in the plane.
        # Wait, p must be in the plane of a,b,c AND equidistant.
        # The equidistant condition: p·a = p·b => p·(a-b) = 0
        # p·a = p·c => p·(a-c) = 0
        # So p is perpendicular to (a-b) and (a-c). The only vector perpendicular to both is (a-b)×(a-c),
        # which is the normal to the plane. But the normal is NOT in the plane (unless degenerate).
        # I made an error. Let me reconsider.
        # Actually, p·(a-b)=0 means p is in the plane that bisects a and b (the great circle equidistant from a and b).
        # Similarly for c. The intersection of these two great circles with the sphere gives two points.
        # One is the circumcenter, the other is its antipode.
        # To find it: p must satisfy p·(a-b)=0 and p·(a-c)=0.
        # These are two linear constraints. The solution space is 1D (a line through origin).
        # So p = t * v where v = (a-b) × (a-c)? No, v = (a-b) × (a-c) is perpendicular to both,
        # so v·(a-b) = 0 and v·(a-c) = 0. Yes! So v = (a-b) × (a-c) satisfies the constraints.
        # But wait, v is the normal to the plane of a,b,c. Is the normal equidistant from a,b,c?
        # v·a = v·b? v·(a-b) = 0, so yes v·a = v·b. Similarly v·a = v·c.
        # So the normal vector IS equidistant from a, b, c! But it's not in the plane...
        # Hmm, but the circumcenter should be in the plane. Let me re-examine.
        # Actually, for a spherical triangle, the circumcenter is NOT necessarily in the plane of the vertices.
        # The circumcenter is the center of the smallest cap containing the triangle.
        # It is the point on the sphere that minimizes the maximum angular distance to a, b, c.
        # For an acute triangle, it's inside; for obtuse, it's outside.
        # The formula: the circumcenter is along the direction of the normal to the plane of a,b,c,
        # specifically p = (a + b + c) / |a + b + c| is NOT correct in general.
        # Let me use a numerical approach: the circumcenter is the point p on the sphere
        # that satisfies p·a = p·b = p·c. This is a system of 3 equations (2 independent + |p|=1).
        # Let's solve it properly.
        # p·a = p·b => p·(a-b) = 0
        # p·a = p·c => p·(a-c) = 0
        # So p is in the null space of [[a-b], [a-c]], which is spanned by (a-b)×(a-c).
        # So p = ±((a-b)×(a-c)) / |((a-b)×(a-c))|
        # Let's verify: does this point have equal dot products with a, b, c?
        # p·a = p·b since p·(a-b)=0. Yes.
        # So the circumcenter is indeed along the normal direction!
        # This makes sense: the set of points equidistant from a and b is a great circle,
        # and the intersection of two such great circles gives two antipodal points,
        # which lie along the normal to the plane of a,b,c.
        
        v = np.cross(a - b, a - c)
        nv = np.linalg.norm(v)
        if nv < 1e-10:
            # Degenerate triangle, skip or use fallback
            continue
        p = v / nv
        # Check which of p or -p gives smaller max distance
        d1 = np.arccos(np.clip(np.max([np.dot(p, a), np.dot(p, b), np.dot(p, c)]), -1, 1))
        d2 = np.arccos(np.clip(np.max([np.dot(-p, a), np.dot(-p, b), np.dot(-p, c)]), -1, 1))
        circumradius = min(d1, d2)
        if circumradius > max_circumradius:
            max_circumradius = circumradius
    return np.degrees(max_circumradius)

def _fibonacci_init(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Initialize with Fibonacci spiral
    pts = _fibonacci_init(N)
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)

    # Iteratively refine: find the facet with largest circumradius,
    # and move one of its vertices to reduce it
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Compute hull and find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        simplices = hull.simplices
        worst_facet = None
        worst_radius = 0.0
        worst_circumcenter = None
        worst_vertex_idx = None
        
        for tri in simplices:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            v = np.cross(a - b, a - c)
            nv = np.linalg.norm(v)
            if nv < 1e-10:
                continue
            p = v / nv
            d1 = np.arccos(np.clip(np.max([np.dot(p, a), np.dot(p, b), np.dot(p, c)]), -1, 1))
            d2 = np.arccos(np.clip(np.max([np.dot(-p, a), np.dot(-p, b), np.dot(-p, c)]), -1, 1))
            circumradius = min(d1, d2)
            if circumradius > worst_radius:
                worst_radius = circumradius
                worst_facet = tri
                # Choose the circumcenter direction that gives the smaller radius
                if d1 <= d2:
                    worst_circumcenter = p
                else:
                    worst_circumcenter = -p
                # Pick the vertex farthest from the circumcenter
                dots = np.array([np.dot(worst_circumcenter, pts[tri[0]]),
                                  np.dot(worst_circumcenter, pts[tri[1]]),
                                  np.dot(worst_circumcenter, pts[tri[2]])])
                worst_vertex_idx = tri[np.argmin(dots)]
        
        if worst_facet is None:
            break
        
        # Move the worst vertex toward the circumcenter
        # Small step in the direction that reduces the circumradius
        step_size = 0.1
        new_pt = pts[worst_vertex_idx] + step_size * (worst_circumcenter - pts[worst_vertex_idx])
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        pts_trial = pts.copy()
        pts_trial[worst_vertex_idx] = new_pt
        trial_radius = _covering_radius_deg(pts_trial)
        
        if trial_radius < best_radius:
            best_radius = trial_radius
            best_pts = pts_trial.copy()
            pts = pts_trial.copy()
        else:
            # Try smaller step or different vertex
            # For now, just break if no improvement after many iterations
            if iteration > 50:
                break
    
    return best_pts
