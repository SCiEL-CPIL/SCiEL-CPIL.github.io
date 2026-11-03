"""Optimized covering of sphere with 50 points using multi-phase approach."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import sys

N = 50
_TIME_LIMIT = 140.0  # safety margin before 150s kill
_DEADLINE = time.time() + _TIME_LIMIT


def _time_remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Compute circumradius of triangle on unit sphere
            a, b, c = tri
            # Side lengths (chord lengths)
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            # Area of triangle
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if area < 1e-15:
                continue
            # Circumradius = (a*b*c) / (4*area) for chord triangle
            cr = (ab * bc * ca) / (4.0 * area)
            # Convert to angular radius: chord = 2*sin(theta/2), so theta = 2*arcsin(chord/2)
            # For circumradius R of triangle with vertices on unit sphere:
            # The angular circumradius satisfies: sin(alpha) = R_chord / 2 * ... 
            # Actually, for points on unit sphere, the circumradius of the spherical triangle
            # can be computed from the chord circumradius:
            # If R is the chord circumradius, the angular radius is arcsin(R/2) * 2... 
            # Let me think more carefully.
            # For a triangle with vertices on unit sphere, the circumcircle in 3D has radius R_chord.
            # The center of this circumcircle is at distance d from origin where R_chord^2 + d^2 = 1 (if center on sphere)
            # Actually the circumcenter may not be on the sphere.
            # The angular distance from the circumcenter (projected to sphere) to each vertex is the covering radius for that facet.
            # Simpler: compute the circumcenter of the 3 points in 3D, project to sphere, measure max angle.
            
            # Compute circumcenter of triangle a,b,c
            # Circumcenter is equidistant from a, b, c
            # Solve: |x-a|^2 = |x-b|^2 = |x-c|^2
            # 2*x.(b-a) = |b|^2 - |a|^2 = 0 (since |a|=|b|=1)
            # 2*x.(c-a) = 0
            # So circumcenter is perpendicular to both (b-a) and (c-a), i.e., parallel to (b-a)x(c-a)
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            # Circumcenter lies along n from origin at some distance t
            # |t*n - a|^2 = R^2, and we want the closest point on sphere to all three vertices
            # Actually for covering radius, we need the point on sphere that maximizes min distance to vertices
            # For a spherical triangle, the circumcenter (if inside) is the point equidistant from all 3 vertices
            # The angular circumradius alpha satisfies: for each vertex v, angle(circumcenter, v) = alpha
            # cos(alpha) = circumcenter . v
            # The circumcenter on sphere: c = n / |n| (if the triangle's circumcenter is on the sphere side)
            # But we need to check: the circumcenter in 3D is at t*n where t = a . n (since |t*n - a|^2 = |a|^2 - 2*t*(a.n) + t^2 = 1 - 2t*(a.n) + t^2, and this should equal R^2 for all vertices)
            # For equidistance: t*n . (b-a) = 0 which is satisfied since n perp (b-a)
            # The distance from circumcenter to each vertex: |t*n - a| = sqrt(1 - 2t*(a.n) + t^2)
            # For this to be the same for a, b, c: a.n = b.n = c.n (which holds since n perp (b-a) and (c-a))
            # So t = a.n works (circumcenter at t*n)
            t = np.dot(a, n)
            circumcenter_3d = t * n
            # Project to sphere
            cc_norm = np.linalg.norm(circumcenter_3d)
            if cc_norm < 1e-15:
                continue
            circumcenter_sphere = circumcenter_3d / cc_norm
            # Angular distance from circumcenter to each vertex
            for v in [a, b, c]:
                cos_ang = np.clip(np.dot(circumcenter_sphere, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                max_cr = max(max_cr, ang)
        return np.degrees(max_cr)
    except Exception:
        return 180.0


def _spherical_lloyd(pts, n_iters=50, tol=1e-8):
    """Spherical Lloyd's algorithm: move each point to centroid of its Voronoi cell."""
    pts = _normalize(pts.copy())
    for it in range(n_iters):
        if _time_remaining() < 5.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (set of hull facets containing it)
        # Move to centroid of the cell's vertices (in 3D), then project to sphere
        new_pts = pts.copy()
        for i in range(N):
            # Find facets containing point i
            cell_points = set()
            for simplex in hull.simplices:
                if i in simplex:
                    for j in simplex:
                        cell_points.add(j)
            if len(cell_points) < 3:
                continue
            cell_arr = np.array([pts[j] for j in cell_points])
            centroid = np.mean(cell_arr, axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 1e-12:
                new_pts[i] = centroid / c_norm
        
        # Check convergence
        diff = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = _normalize(new_pts)
        if diff < tol:
            break
    
    return pts


def _simulated_annealing(pts, n_iters=2000, t_start=0.5, t_end=0.01, step_size=0.1):
    """Simulated annealing to escape local minima."""
    pts = _normalize(pts.copy())
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    for it in range(n_iters):
        if _time_remaining() < 5.0:
            break
        # Annealing schedule
        t = t_start * (t_end / t_start) ** (it / max(n_iters - 1, 1))
        
        # Random perturbation
        idx = np.random.randint(N)
        # Random direction
        d = np.random.randn(3)
        d /= np.linalg.norm(d)
        
        new_pts = pts.copy()
        new_pts[idx] += step_size * t * d
        new_pts = _normalize(new_pts)
        
        new_cr = _covering_radius(new_pts)
        
        if new_cr < current_cr or np.random.random() < np.exp((current_cr - new_cr) / max(t, 1e-10)):
            current_cr = new_cr
            pts = new_pts
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = new_pts.copy()
    
    return best_pts


def _local_polish(pts, n_iters=100, tol=1e-10):
    """High-precision local optimization using gradient-free method."""
    pts = _normalize(pts.copy())
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    step = 0.05
    for it in range(n_iters):
        if _time_remaining() < 3.0:
            break
        improved = False
        for i in range(N):
            if _time_remaining() < 3.0:
                break
            # Try small moves in random directions
            for _ in range(10):
                d = np.random.randn(3)
                d /= np.linalg.norm(d)
                new_pts = pts.copy()
                new_pts[i] += step * d
                new_pts = _normalize(new_pts)
                new_cr = _covering_radius(new_pts)
                if new_cr < current_cr:
                    current_cr = new_cr
                    pts = new_pts
                    improved = True
                    if new_cr < best_cr:
                        best_cr = new_cr
                        best_pts = new_pts.copy()
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    
    return best_pts


def _fibonacci_points():
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    """Random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)


def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    np.random.seed(42)
    
    # Phase 1: Start with Fibonacci spiral, run Lloyd's
    pts = _fibonacci_points()
    pts = _spherical_lloyd(pts, n_iters=30)
    cr1 = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = cr1
    
    # Phase 2: Simulated annealing from multiple starts
    starts = [
        _fibonacci_points(),
        _random_points(seed=1),
        _random_points(seed=2),
        _random_points(seed=3),
    ]
    
    for s_idx, start in enumerate(starts):
        if _time_remaining() < 15.0:
            break
        
        # Lloyd's first
        pts = _spherical_lloyd(start, n_iters=20)
        cr = _covering_radius(pts)
        
        # SA to escape local minima
        pts = _simulated_annealing(pts, n_iters=1500, t_start=0.3, t_end=0.005, step_size=0.15)
        cr = _covering_radius(pts)
        
        # Lloyd's again to settle
        pts = _spherical_lloyd(pts, n_iters=30)
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Phase 3: Final polish
    if _time_remaining() > 10.0:
        best_pts = _local_polish(best_pts, n_iters=50)
        best_cr = _covering_radius(best_pts)
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    
    # Final validation: check no coincident points, not in closed hemisphere
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10):
        best_pts = _fibonacci_points()
    
    # Check not in closed hemisphere: origin should be inside convex hull
    try:
        hull = ConvexHull(best_pts)
        # Check if origin is inside by verifying all facet normals point outward
        inside = True
        for simplex in hull.simplices:
            tri = best_pts[simplex]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-15:
                n = n / n_norm
                if np.dot(n, tri[0]) < -1e-10:
                    inside = False
                    break
        if not inside:
            # Fallback to Fibonacci
            best_pts = _fibonacci_points()
    except Exception:
        best_pts = _fibonacci_points()
    
    return best_pts
