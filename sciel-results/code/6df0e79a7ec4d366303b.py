"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
import scipy.optimize

# Constants
_DEADLINE = 145.0  # Safety margin before 150s kill

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _covering_radius_exact(pts):
    """
    Calculate the exact covering radius by finding the circumradius of all 
    Delaunay tetrahedra (facets of the convex hull in 3D are triangles, 
    but for sphere covering, the 'holes' are defined by the Delaunay triangulation 
    on the sphere surface, which corresponds to the tetrahedra in the 3D Delaunay 
    tessellation if we consider the convex hull, or more accurately, the spherical 
    Delaunay triangulation).
    
    Actually, for points on a sphere, the covering radius is the maximum circumradius 
    of the spherical triangles formed by the Delaunay triangulation of the points 
    on the sphere. 
    However, scipy.spatial.Delaunay computes the 3D Delaunay triangulation. 
    The facets of the convex hull are triangles. The interior of the convex hull 
    is filled with tetrahedra. 
    The covering radius of the sphere is determined by the largest empty spherical 
    triangle. This corresponds to the circumcircles of the faces of the convex hull? 
    No. The covering radius is the max distance from any point on the sphere to the 
    nearest site. This occurs at the circumcenters of the Delaunay triangles 
    (spherical). 
    
    In 3D, the Delaunay triangulation of points on a sphere includes the convex hull 
    faces. The circumcenters of the convex hull faces (triangles) are NOT the covering 
    points unless the tetrahedra are empty. 
    Actually, the Voronoi cells of the points on the sphere are bounded by arcs of 
    great circles. The vertices of these Voronoi cells are the circumcenters of the 
    Delaunay triangles. 
    The 3D Delaunay triangulation's tetrahedra correspond to the Voronoi vertices. 
    Specifically, each tetrahedron in the 3D Delaunay triangulation has a circumcenter. 
    If that circumcenter is outside the convex hull (which it is for hull faces) or 
    inside, it defines a Voronoi vertex. 
    The covering radius is the maximum distance from any such circumcenter (projected 
    to the sphere? No, the circumcenter of a spherical triangle lies on the sphere 
    if the triangle is small, but generally the circumcenter of 3 points on a sphere 
    is the center of the circle passing through them. If the circle is a small circle, 
    the center is not on the sphere. The "point on the sphere" farthest from the 3 
    sites is the pole of the great circle? No.
    
    Let's stick to the definition: Covering radius R is min R such that balls of 
    radius R cover the sphere. 
    The critical points are the circumcenters of the Delaunay triangles (spherical). 
    For 3 points on a unit sphere, the circumcenter of the spherical triangle is the 
    point on the sphere equidistant from the 3 vertices. 
    This point is the normalized cross product of (p1-p2) x (p1-p3)? No, that's the 
    normal to the plane. The spherical circumcenter is the intersection of the sphere 
    with the line passing through the origin and the Euclidean circumcenter of the 
    3 points? 
    Yes. The Euclidean circumcenter of 3 points on a sphere lies on the line 
    connecting the sphere center (origin) to the spherical circumcenter. 
    So, for each face of the convex hull (which are the Delaunay triangles on the 
    sphere surface), we can compute the Euclidean circumcenter. 
    The distance from this Euclidean circumcenter to the origin is 1/sin(theta), where 
    theta is the spherical circumradius? 
    Actually, if C is the Euclidean circumcenter of triangle ABC, and O is the origin, 
    then the spherical circumradius r is the angle AOC. 
    Since C lies on the normal to the plane ABC, and |A|=|B|=|C|=1, 
    the projection of A onto the line OC has length cos(r). 
    So |C| * cos(r) = 1? No. 
    Let's just compute the angle. 
    For a face with vertices p1, p2, p3:
    1. Compute normal n = (p2-p1) x (p3-p1).
    2. The Euclidean circumcenter C lies on the line t*n. 
    3. We want C such that |C - p1| = |C - p2|. 
    4. |t*n - p1|^2 = |t*n - p2|^2 => -2t*n.p1 + |p1|^2 = -2t*n.p2 + |p2|^2 => 2t*n.(p2-p1) = 0. This is always true if n is perp to p2-p1. 
    5. We need |C - p1| = R_euc. The distance from C to the sphere surface along the radial direction is not the metric. 
    The covering radius is the angular distance. 
    The point on the sphere closest to C (Euclidean) is C / |C|. 
    The angular distance from p1 to C/|C| is arccos( p1 . (C/|C|) ). 
    Since C is the circumcenter, p1 . C = p2 . C = p3 . C = k. 
    So p1 . (C/|C|) = k / |C|. 
    So the angular distance is arccos( k / |C| ). 
    
    We need to identify which faces are the "empty" caps. 
    In the Delaunay triangulation of points on a sphere, the convex hull faces 
    are the Delaunay triangles. The circumcenter of such a triangle must be "outside" 
    the convex hull in the sense that the sphere is locally contained in the cap. 
    Actually, for points on a sphere, ALL convex hull faces correspond to Delaunay 
    triangles. The circumcenter of a convex hull face lies OUTSIDE the convex hull 
    (on the side of the sphere). 
    So we can just take all faces of the convex hull, compute their circumcenters, 
    and find the max angular distance to the vertices.
    """
    try:
        hull = scipy.spatial.ConvexHull(pts)
        facets = hull.simplices
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        # Use a simple sampling estimate
        samples = _generate_samples(5000)
        dots = samples @ pts.T
        max_dots = dots.max(axis=1)
        min_max_dot = np.min(max_dots)
        return np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
    
    max_angle = 0.0
    for face in facets:
        p1 = pts[face[0]]
        p2 = pts[face[1]]
        p3 = pts[face[2]]
        
        # Compute Euclidean circumcenter
        # C = t * n
        # n = (p2-p1) x (p3-p1)
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Solve for t such that |t*n - p1|^2 = |t*n - p2|^2 is not enough, 
        # we need the circumcenter of the triangle in 3D space.
        # The circumcenter C satisfies |C-p1|^2 = |C-p2|^2 = |C-p3|^2.
        # Since C is on the line t*n, we can find t by projecting p1 onto n.
        # Wait, the circumcenter of a triangle in 3D is unique.
        # C = t * n.
        # |t*n - p1|^2 = |t*n|^2 - 2t*n.p1 + |p1|^2
        # |t*n - p2|^2 = |t*n|^2 - 2t*n.p2 + |p2|^2
        # Since |p1|=|p2|=1, n.p1 = n.p2? No.
        # n is perpendicular to p2-p1, so n.p2 - n.p1 = n.(p2-p1) = 0. 
        # So n.p1 = n.p2 = n.p3. Let this value be c.
        c = np.dot(n, p1)
        
        # We want C such that it is equidistant to p1, p2, p3.
        # This is automatically satisfied for any t if n.p1=n.p2=n.p3? 
        # No, the distance squared is t^2 - 2tc + 1. 
        # This is the same for all 3 points. So ANY point on the line t*n is equidistant 
        # to p1, p2, p3? 
        # No, that's the axis of rotation. The circumcenter is the specific point on 
        # this axis that is in the plane of the triangle? 
        # No, the circumcenter of a triangle in 3D is the intersection of the 
        # perpendicular bisector planes. 
        # The perpendicular bisector of p1p2 contains the line t*n? 
        # Midpoint M12 = (p1+p2)/2. Vector p2-p1. 
        # The plane is { x | (x-M12).(p2-p1) = 0 }.
        # Does t*n satisfy this? (t*n - (p1+p2)/2).(p2-p1) = t*n.(p2-p1) - (p1+p2)/2.(p2-p1) = 0 - (|p2|^2 - |p1|^2)/2 = 0. 
        # Yes, the entire line t*n is in the perpendicular bisector plane of p1p2. 
        # Similarly for p2p3 and p1p3. 
        # So the entire line t*n is the set of points equidistant from p1, p2, p3. 
        # This is the axis of the circular cylinder through p1, p2, p3. 
        # The "circumcenter" of the spherical triangle is the point on the sphere 
        # on this axis. 
        # There are two such points: n and -n. 
        # We want the one that is the center of the "empty" cap. 
        # For a convex hull face, the normal n points OUTWARDS. 
        # The cap is centered at n (normalized). 
        # The angular distance from p1 to n is arccos( p1 . n ). 
        # Since n is the unit normal, p1 . n = c. 
        # So the angular distance is arccos(c). 
        
        angle = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        if angle > max_angle:
            max_angle = angle
            
    return max_angle

def _generate_samples(n_samples):
    """Generate sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples

def _optimize_with_delaunay(pts_init, time_limit, start_time):
    """
    Optimize points by minimizing the max circumradius of convex hull facets.
    Uses L-BFGS-B.
    """
    current_time = time.time()
    if current_time - start_time > time_limit:
        return pts_init, _covering_radius_exact(pts_init)
    
    def objective_and_grad(x):
        current_time = time.time()
        if current_time - start_time > time_limit:
            return 1e6, np.zeros_like(x)
        
        pts = _normalize(x.reshape(N, 3))
        
        try:
            hull = scipy.spatial.ConvexHull(pts)
            facets = hull.simplices
        except Exception:
            return 1e6, np.zeros_like(x)
            
        max_radius = 0.0
        # Identify the worst facet
        worst_facet = None
        worst_normal = None
        worst_cos = 1.0 # cos(angle), we want to minimize angle, so maximize cos? 
        # Wait, we want to MINIMIZE the covering radius (angle). 
        # So we want to MAXIMIZE the minimum cos(angle) over all facets? 
        # No, we want to minimize the MAX angle. 
        # So we find the facet with the largest angle (smallest cos). 
        # And we want to increase that cos (move points closer to the normal).
        
        min_cos = 1.0
        for face in facets:
            p1 = pts[face[0]]
            p2 = pts[face[1]]
            p3 = pts[face[2]]
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            c = np.dot(n, p1)
            if c < min_cos:
                min_cos = c
                worst_facet = face
                worst_normal = n
                max_radius = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
                
        if worst_facet is None:
            return 0.0, np.zeros_like(x)
            
        # Gradient: 
        # We want to increase c = n . p1. 
        # n depends on p1, p2, p3. 
        # This is complex. 
        # Alternative: Use finite differences? Too slow.
        # Approximation: Move p1, p2, p3 towards the normal n. 
        # This is a heuristic gradient.
        
        grad = np.zeros_like(x)
        # Move points in the worst facet towards the normal
        # n is the direction to move TO. 
        # The gradient of the distance to n is roughly -n for each point?
        # If we move p1 towards n, the triangle becomes "flatter" towards the pole? 
        # Actually, moving the vertices towards the pole (n) increases the angular 
        # distance from the pole to the vertices? No, it DECREASES the radius of the 
        # cap centered at n. 
        # Wait, the cap is centered at n. The radius is the distance from n to the 
        # vertices. 
        # If we move vertices TOWARDS n, the distance decreases. 
        # So we should move p1, p2, p3 in the direction of n.
        
        for idx in worst_facet:
            grad[idx*3:(idx+1)*3] = -worst_normal # Gradient of distance to n is (p-n)/|p-n|, approx -n if p is close to n? 
            # Actually, to minimize distance to n, we move towards n. 
            # So the update should be p += lr * (n - p). 
            # In gradient descent, we do p -= lr * grad. 
            # So grad should be (p - n). 
            p = pts[idx]
            grad[idx*3:(idx+1)*3] = p - worst_normal
            
        return max_radius, grad.flatten()

    # Flatten points for optimization
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B
    # Since we have a non-smooth objective (max of smooth functions), 
    # and the gradient is heuristic, we might use a simple line search or 
    # just run a few steps of gradient descent with the heuristic gradient.
    
    # Let's try scipy.optimize.minimize
    try:
        res = scipy.optimize.minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        if res.success or res.fun < _covering_radius_exact(pts_init):
            pts_opt = _normalize(res.x.reshape(N, 3))
            return pts_opt, _covering_radius_exact(pts_opt)
    except Exception:
        pass
        
    return pts_init, _covering_radius_exact(pts_init)

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses Fibonacci spiral initialization + Delaunay-based local optimization.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts_init = _normalize(pts_init)
    
    best_pts = pts_init.copy()
    best_radius = _covering_radius_exact(pts_init)
    
    rng = np.random.default_rng(42)
    
    # Try multiple random perturbations
    n_attempts = 20 # Max attempts, but limited by time
    
    for attempt in range(n_attempts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10: # Leave 10s for safety
            break
        
        if attempt == 0:
            current_pts = pts_init
        else:
            # Perturb the best points found so far
            noise = rng.normal(0, 0.05, size=best_pts.shape)
            current_pts = _normalize(best_pts + noise)
        
        # Run optimization
        remaining = deadline - (time.time() - start_time) - 2.0
        if remaining < 1.0:
            break
            
        opt_pts, opt_radius = _optimize_with_delaunay(current_pts, remaining, start_time)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
            
    # Final polish: try to refine the best points with a tighter optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        remaining = deadline - elapsed - 1.0
        if remaining > 1.0:
            # Use a smaller perturbation or just re-optimize
            final_pts, final_radius = _optimize_with_delaunay(best_pts, remaining, start_time)
            if final_radius < best_radius:
                best_pts = final_pts.copy()
                best_radius = final_radius
                
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
