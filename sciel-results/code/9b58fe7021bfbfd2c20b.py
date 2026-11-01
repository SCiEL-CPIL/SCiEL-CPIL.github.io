"""Rewritten program: optimize 50-point sphere covering via Lloyd + annealing."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius in degrees for points on unit sphere.
    Uses convex hull facets: for each facet, compute circumcenter of the
    facet (in 3D), check if the cap is empty (circumcenter direction is
    not blocked by other points), and compute the angular distance from
    the facet's circumcenter direction to the nearest point."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0  # degenerate
    
    facets = hull.simplices  # shape (F, 3)
    max_ang = 0.0
    
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        # Compute circumcenter of the triangle on the unit sphere.
        # The circumcenter of a spherical triangle lies along the direction
        # perpendicular to the plane of the triangle, at the appropriate distance.
        # For a triangle with vertices on the unit sphere, the 3D circumcenter
        # (center of the circumscribed circle) is:
        #   O = (a + b + c) / (1 + a·b + b·c + c·a)  ... no, that's not right.
        
        # Better approach: The circumcenter of triangle abc in 3D (center of the
        # circle passing through a, b, c) lies on the line through the centroid
        # perpendicular to the plane. For points on the unit sphere, the 
        # circumcenter of the great-circle triangle is in the direction of
        # the cross product based formula.
        
        # Actually, for the covering radius, we need: for each facet, find the
        # point on the unit sphere that is equidistant from the three vertices
        # of the facet (the spherical circumcenter), and check that no other
        # point is closer to it. The angular distance from that circumcenter to
        # the nearest point is the circumradius of that facet.
        
        # Spherical circumcenter: direction proportional to
        #   a*(b×c) + b*(c×a) + c*(a×b)  ... let me use a different formula.
        
        # The circumcenter of a spherical triangle with vertices a, b, c (unit vectors)
        # is in the direction:
        #   d = (a × b) × (b × c) + (b × c) × (c × a) + (c × a) × (a × b)
        # Hmm, let me use the standard formula.
        
        # The plane through a, b, c has normal n = (b-a) × (c-a).
        # The circumcenter in 3D (of the circle through a,b,c) is:
        #   O = a + t * n  where t is chosen so |O - a| = |O - b| = |O - c|
        # Since |a|=|b|=|c|=1, we have |O|^2 - 2 O·a + 1 = |O|^2 - 2 O·b + 1
        # => O·a = O·b = O·c = k (say)
        # So O = k * (a + b + c) / (a·(a+b+c) + ... ) ... 
        
        # Simpler: the circumcenter of the triangle in 3D is:
        #   O = (a*(|b|^2|c|^2 - (a·b)^2) + ...) -- too complex.
        
        # Let me just use: the point on the unit sphere equidistant from a, b, c.
        # This is the normalized version of the 3D circumcenter of triangle abc.
        # 3D circumcenter: solve for O such that |O-a|=|O-b|=|O-c|.
        # |O|^2 - 2O·a = |O|^2 - 2O·b => O·(a-b) = 0
        # |O|^2 - 2O·b = |O|^2 - 2O·c => O·(b-c) = 0
        # So O is perpendicular to (a-b) and (b-c), i.e., O is parallel to (a-b)×(b-c).
        # And O also must satisfy O·a = O·b = O·c.
        # Let n = (a-b) × (b-c). Then O = alpha * n.
        # O·a = O·b => alpha*(n·a - n·b) = 0, which is automatically satisfied
        # since n is perpendicular to a-b... wait, n = (a-b)×(b-c), so n·(a-b)=0 and n·(b-c)=0.
        # n·a - n·b = n·(a-b) = 0. Good.
        # n·b - n·c = n·(b-c) = 0. Good.
        # So O = alpha * n works. To get the circumcenter on the unit sphere,
        # we just normalize n. But we need to pick the right sign (the one
        # that gives the circumcenter on the same side as the facet).
        
        n = np.cross(a - b, b - c)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        d = n / n_norm  # direction of circumcenter
        
        # Check which side: the circumcenter should be on the outward side of the facet.
        # The facet normal (outward) is:
        fn = np.cross(b - a, c - a)
        fn_norm = np.linalg.norm(fn)
        if fn_norm < 1e-12:
            continue
        fn = fn / fn_norm
        
        # The circumcenter direction should have positive dot with the facet normal
        # (outward from convex hull). If not, flip.
        if np.dot(d, fn) < 0:
            d = -d
        
        # Now d is the direction of the spherical circumcenter.
        # Check if the cap is empty: no other point should be closer to d than
        # the three vertices are.
        # The angular distance from d to a (or b or c) is the circumradius of this facet.
        dot_a = np.clip(np.dot(d, a), -1.0, 1.0)
        ang_radius = np.degrees(np.arccos(dot_a))
        
        # Check if any other point is closer to d
        dots = np.clip(pts @ d, -1.0, 1.0)
        min_ang_other = np.degrees(np.arccos(dots.max()))
        
        # The covering radius contribution from this facet is ang_radius
        # (the cap is empty if the three vertices are the closest, which they
        # should be for a proper Voronoi facet, but let's just use ang_radius)
        if ang_radius > max_ang:
            max_ang = ang_radius
    
    return max_ang


def _lloyd_step(pts, n_iter=20):
    """Lloyd iteration: move each point toward its Voronoi cell's worst circumcenter."""
    pts = _normalize(pts)
    for _ in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (facets adjacent to it)
        # and compute the worst circumcenter (the one with largest circumradius).
        # Then move the point toward that circumcenter.
        
        # Build point-to-facet mapping
        point_facets = [[] for _ in range(N)]
        for i, tri in enumerate(hull.simplices):
            for p in tri:
                point_facets[p].append(i)
        
        new_pts = pts.copy()
        for p in range(N):
            if not point_facets[p]:
                continue
            
            worst_dir = None
            worst_radius = -1.0
            
            for fi in point_facets[p]:
                tri = hull.simplices[fi]
                a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                
                n = np.cross(a - b, b - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                d = n / n_norm
                
                fn = np.cross(b - a, c - a)
                fn_norm = np.linalg.norm(fn)
                if fn_norm < 1e-12:
                    continue
                fn = fn / fn_norm
                
                if np.dot(d, fn) < 0:
                    d = -d
                
                dot_a = np.clip(np.dot(d, a), -1.0, 1.0)
                ang_radius = np.degrees(np.arccos(dot_a))
                
                if ang_radius > worst_radius:
                    worst_radius = ang_radius
                    worst_dir = d
            
            if worst_dir is not None:
                # Move toward worst circumcenter with a small step
                step = 0.3
                new_pts[p] = pts[p] + step * (worst_dir - pts[p])
        
        new_pts = _normalize(new_pts)
        # Check improvement
        if np.allclose(new_pts, pts, atol=1e-10):
            break
        pts = new_pts
    
    return pts


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on unit sphere using golden angle spiral with offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _optimize(pts, deadline, max_lloyd_iters=30):
    """Run Lloyd iterations until deadline."""
    pts = _normalize(pts)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    while time.time() < deadline:
        pts = _lloyd_step(pts, n_iter=1)
        cr = _covering_radius(pts)
        if cr < best_cr - 1e-9:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_cr = 180.0
    
    # Try multiple starting configurations
    starts = []
    # Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    # Random starts
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
    # Slightly perturbed Fibonacci
    for _ in range(3):
        base = _fibonacci_sphere(N)
        noise = rng.standard_normal((N, 3)) * 0.1
        starts.append(_normalize(base + noise))
    
    for s in starts:
        if time.time() > deadline:
            break
        
        # Quick Lloyd optimization
        pts, cr = _optimize(s, deadline)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
