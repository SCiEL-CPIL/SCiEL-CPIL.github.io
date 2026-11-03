"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    """
    Compute the covering radius in radians using convex hull facet circumradii.
    The covering radius is the maximum over all facets of the angular radius
    of the smallest spherical cap containing that facet.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use sampled approach
        rng = np.random.RandomState(0)
        theta = np.linspace(0, np.pi, 80)
        phi = np.linspace(0, 2*np.pi, 160)
        theta, phi = np.meshgrid(theta, phi)
        y = np.column_stack([
            np.sin(theta)*np.cos(phi).ravel(),
            np.sin(theta)*np.sin(phi).ravel(),
            np.cos(theta).ravel()
        ])
        dots = y @ pts.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        min_angles = angles.min(axis=1)
        return np.max(min_angles)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        if len(simplex) == 3:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute circumradius of spherical triangle
            # Side lengths
            ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            # Use formula for circumradius of spherical triangle
            s = (ab + bc + ca) / 2.0
            sin_a = np.sin(ab)
            sin_b = np.sin(bc)
            sin_c = np.sin(ca)
            sin_s = np.sin(s)
            cos_sa = np.cos(s - ab)
            cos_sb = np.cos(s - bc)
            cos_sc = np.cos(s - ca)
            
            denom = 4.0 * sin_s * cos_sa * cos_sb * cos_sc
            if abs(denom) < 1e-15:
                # Degenerate case
                R = max(ab, bc, ca) / 2.0
            else:
                sin_R = (sin_a * sin_b * sin_c) / denom
                sin_R = np.clip(sin_R, -1.0, 1.0)
                R = np.arcsin(sin_R)
            
            max_radius = max(max_radius, R)
    
    return max_radius

def _facet_circumdata(pts):
    """
    For each triangular facet of the convex hull, compute:
      - circumradius (angular)
      - circumcenter (unit vector)
      - the three vertex indices
    Returns list of (radius, center, simplex_indices).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Compute circumcenter of the spherical triangle
        # The circumcenter is the point equidistant from a, b, c on the sphere.
        # It lies along the normal to the plane of the triangle, projected onto the sphere.
        # For a triangle on the unit sphere, the circumcenter is in the direction of
        # the vector that is equidistant from all three vertices.
        
        # Method: solve for the point p such that |p-a| = |p-b| = |p-c| and |p|=1
        # This means p is in the direction of the circumcenter of the planar triangle
        # formed by projecting a,b,c to a plane, but more directly:
        
        # The circumcenter of a spherical triangle is along the direction:
        # n = (a x b) x (b x c) ... no, let's use the linear system.
        
        # For unit vectors a, b, c, the circumcenter p satisfies:
        # p . a = p . b = p . c = cos(R)
        # and |p| = 1
        # So p is in the span of a, b, c, and we can write p = alpha*a + beta*b + gamma*c
        # with the constraints p.a = p.b = p.c = cos(R)
        
        # Let M be the 3x3 matrix with rows a, b, c (as unit vectors)
        # Then p.M^T = [cos(R), cos(R), cos(R)]^T ... wait
        # p.a = cos(R), p.b = cos(R), p.c = cos(R)
        # So M @ p = cos(R) * ones(3)
        # where M = [[a], [b], [c]]
        
        M = np.array([a, b, c])
        rhs = np.ones(3)
        
        try:
            # Solve M @ p = rhs
            # But p is a 3D vector, M is 3x3
            p = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Degenerate
            continue
        
        pn = np.linalg.norm(p)
        if pn < 1e-12:
            continue
        p = p / pn
        
        # Verify: p should be equidistant from a, b, c
        cos_R = np.dot(p, a)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check that p is on the correct side (facing outward)
        # The circumcenter should be such that the cap is "empty" - i.e., no other
        # points are inside the cap. For now, we just compute the geometric circumcenter.
        # We need to ensure it's the one that gives the covering radius, not the antipodal one.
        
        # Actually, there are two circumcenters (antipodal). We want the one that
        # is "outside" the convex hull in the sense that the facet is a face.
        # The normal to the facet points outward. The circumcenter for the covering
        # radius should be on the same side as the outward normal.
        
        # Compute outward normal
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v = v / vn
        
        # The outward normal for a convex hull facet points away from the origin
        # (since we're on the unit sphere, the hull contains the origin in its interior
        # if the points are in general position and not all in a hemisphere)
        # Actually, for points on the unit sphere, the convex hull contains the origin
        # if and only if the points are not all in a closed hemisphere.
        # The outward normal of a facet should point away from the interior of the hull.
        # Since the hull contains the origin (for a good covering), the outward normal
        # points away from the origin, i.e., dot(n, centroid_of_facet) > 0.
        
        centroid = (a + b + c) / 3.0
        if np.dot(v, centroid) < 0:
            v = -v
        
        # The circumcenter p should be on the same side as the outward normal
        if np.dot(p, v) < 0:
            p = -p
            R = np.arccos(np.clip(-cos_R, -1.0, 1.0))
        
        out.append((R, p, simplex))
    
    return out

def _worst_facet_shrink(pts, alpha=0.1, deadline=None):
    """
    Identify the facet with the largest circumradius and move its vertices
    toward the circumcenter along the great circle.
    """
    pts = _normalize(pts)
    facets = _facet_circumdata(pts)
    if not facets:
        return pts, _covering_radius(pts)
    
    best = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(500):
        if deadline is not None and time.time() > deadline:
            break
        
        facets = _facet_circumdata(pts)
        if not facets:
            break
        
        # Find the facet with the largest circumradius
        max_idx = np.argmax([f[0] for f in facets])
        max_R, center, simplex = facets[max_idx]
        
        # Move the three vertices toward the circumcenter
        new_pts = pts.copy()
        for idx in simplex:
            p = pts[idx]
            # Move along the great circle from p toward center
            # Use spherical linear interpolation (slerp)
            cos_theta = np.clip(np.dot(p, center), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            if theta < 1e-10:
                continue
            sin_theta = np.sin(theta)
            # Slerp: p(t) = (sin((1-t)*theta)/sin(theta)) * p + (sin(t*theta)/sin(theta)) * center
            t = alpha
            w1 = np.sin((1-t)*theta) / sin_theta
            w2 = np.sin(t*theta) / sin_theta
            new_p = w1 * p + w2 * center
            new_pts[idx] = new_p
        
        new_pts = _normalize(new_pts)
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr - 1e-14:
            pts = new_pts
            best = new_pts.copy()
            best_cr = new_cr
        else:
            # Try smaller step
            alpha = alpha * 0.7
            if alpha < 1e-6:
                break
    
    return best, best_cr

def _fibonacci_points(n, seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_based_points(n, seed):
    """Generate points based on icosahedron vertices with some perturbation."""
    rng = np.random.RandomState(seed)
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Start with icosahedron vertices and add more points
    pts = [verts]
    while len(pts) < n:
        # Add random points
        new_pts = rng.randn(1, 3)
        new_pts = _normalize(new_pts)
        pts.append(new_pts)
    
    pts = np.vstack(pts[:n])
    return _normalize(pts)

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius (log-sum-exp over facet radii) and its
    gradient with respect to the input points.
    """
    points = _normalize(points)
    facets = _facet_circumdata(points)
    if not facets:
        return 0.0, np.zeros_like(points)
    
    radii = np.array([f[0] for f in facets])
    centers = np.array([f[1] for f in facets])
    simplices = [f[2] for f in facets]
    
    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta
    
    w = np.exp(beta * (radii - m))
    w /= w.sum()
    
    grad = np.zeros_like(points)
    for i in range(len(radii)):
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan
    
    return val, grad

def _lbfgs_refine(points, max_time=30.0, start_time=None, beta=50.0):
    """Refine points using L-BFGS-B on a smooth log-sum-exp surrogate."""
    from scipy.optimize import minimize
    
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    
    if start_time is None:
        start_time = time.time()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
        )
        
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)
        
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass
    
    return best, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points(N, seed=0))
    
    # Random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_random_points(N, seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points(N, seed=10))
    
    best_pts = initializations[0]
    best_cr = _covering_radius(best_pts)
    
    # Optimize from each start
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        # Check if this start is already better
        cr_init = _covering_radius(init_pts)
        if cr_init < best_cr:
            best_pts = init_pts.copy()
            best_cr = cr_init
        
        # First try worst-facet shrinkage (deterministic, geometric)
        try:
            opt_pts, opt_cr = _worst_facet_shrink(init_pts, alpha=0.1, deadline=deadline)
            if opt_cr < best_cr:
                best_pts = opt_pts
                best_cr = opt_cr
        except Exception:
            pass
        
        # Then try L-BFGS refinement on smooth surrogate
        if time.time() < deadline - 10.0:
            try:
                opt_pts, opt_cr = _lbfgs_refine(best_pts, max_time=20.0, start_time=start_time)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
    
    # Final polish with tighter tolerance if time allows
    if time.time() < deadline - 10.0:
        try:
            opt_pts, opt_cr = _lbfgs_refine(best_pts, max_time=20.0, start_time=start_time, beta=100.0)
            if opt_cr < best_cr:
                best_pts = opt_pts
                best_cr = opt_cr
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
