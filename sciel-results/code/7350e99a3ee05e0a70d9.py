"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    pts = np.asarray(pts, dtype=float)
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
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 200)
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
            # Compute circumcenter of the spherical triangle
            # The circumcenter is along the normal to the plane of the triangle
            v = np.cross(b - a, c - a)
            vn = np.linalg.norm(v)
            if vn < 1e-12:
                continue
            v = v / vn
            
            # Ensure outward normal (away from origin for convex hull containing origin)
            centroid = (a + b + c) / 3.0
            if np.dot(v, centroid) < 0:
                v = -v
            
            # Circumradius is angle between circumcenter and any vertex
            cos_R = np.clip(np.dot(v, a), -1.0, 1.0)
            R = np.arccos(cos_R)
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
        
        # Compute circumcenter using the linear system approach
        # p . a = p . b = p . c = cos(R)
        M = np.array([a, b, c])
        rhs = np.ones(3)
        
        try:
            p = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Fallback to geometric normal
            v = np.cross(b - a, c - a)
            vn = np.linalg.norm(v)
            if vn < 1e-12:
                continue
            p = v / vn
        
        pn = np.linalg.norm(p)
        if pn < 1e-12:
            continue
        p = p / pn
        
        # Verify: p should be equidistant from a, b, c
        cos_R = np.dot(p, a)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check that p is on the correct side (facing outward)
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v = v / vn
        
        centroid = (a + b + c) / 3.0
        if np.dot(v, centroid) < 0:
            v = -v
        
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
            cos_theta = np.clip(np.dot(p, center), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            if theta < 1e-10:
                continue
            sin_theta = np.sin(theta)
            # Slerp
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
    
    # Numerically stable LSE
    exps = np.exp(beta * (radii - m))
    lse = m + np.log(np.sum(exps)) / beta
    w = exps / np.sum(exps)
    
    grad = np.zeros_like(points)
    for i in range(len(radii)):
        wi = w[i]
        if wi < 1e-15:
            continue
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            # Gradient of arccos(p.center) wrt p is -center/sin_r
            g = -center / sin_r
            # Project to tangent plane at p
            g_tan = g - np.dot(g, p) * p
            grad[idx] += wi * g_tan
    
    return lse, grad

def _riesz_energy_grad(points, s=12.0):
    """
    Riesz s-energy (sum over pairs of angular distance^{-s}) and gradient (tangent-projected).
    Used as an auxiliary smoother to diversify local minima.
    """
    points = _normalize(points)
    n = len(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)
    sins = np.sin(angles)
    sins = np.maximum(sins, 1e-15)

    ang_powers = angles ** (-s - 1)
    np.fill_diagonal(ang_powers, 0.0)
    np.fill_diagonal(sins, 1.0)
    np.fill_diagonal(dots, 1.0)

    weights = s * ang_powers / sins
    np.fill_diagonal(weights, 0.0)

    grad = weights @ points
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points

    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(angles[iu] ** (-s)))
    return energy, grad_tan

def _lbfgs_refine(points, max_time=30.0, start_time=None, beta=50.0, deadline=None):
    """Refine points using L-BFGS-B on a smooth log-sum-exp surrogate."""
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    
    if start_time is None:
        start_time = time.time()
    
    def objective_and_grad(x_flat):
        if deadline is not None and time.time() > deadline:
            raise Exception("timeout")
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

def _lbfgs_riesz(points, s=12.0, maxiter=200, deadline=None):
    """
    L-BFGS-B on Riesz energy (auxiliary smoother).
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        if deadline is not None and time.time() > deadline:
            raise Exception("timeout")
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _riesz_energy_grad(x, s=s)
        # grad already tangent-projected
        return val, grad.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": 1e-12,
                "gtol": 1e-10,
                "maxfun": 400,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (limit is 120s, evaluator kills at 150s)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points(N, seed=0))
    
    # Random starts with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
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
        
        # 1. Worst-facet shrinkage (deterministic, geometric)
        if time.time() < deadline - 20.0:
            try:
                opt_pts, opt_cr = _worst_facet_shrink(init_pts, alpha=0.1, deadline=deadline - 10.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # 2. L-BFGS refinement on smooth surrogate (LSE)
        if time.time() < deadline - 15.0:
            try:
                opt_pts, opt_cr = _lbfgs_refine(best_pts, max_time=15.0, start_time=start_time, beta=50.0, deadline=deadline - 5.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # 3. Riesz energy refinement (auxiliary smoother)
        if time.time() < deadline - 10.0:
            try:
                opt_pts, opt_cr = _lbfgs_riesz(best_pts, s=12.0, maxiter=150, deadline=deadline - 5.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
    
    # Final polish with tighter tolerance if time allows
    if time.time() < deadline - 10.0:
        try:
            # High beta LSE polish
            opt_pts, opt_cr = _lbfgs_refine(best_pts, max_time=15.0, start_time=start_time, beta=100.0, deadline=deadline - 5.0)
            if opt_cr < best_cr:
                best_pts = opt_pts
                best_cr = opt_cr
        except Exception:
            pass
        
        # Another Riesz polish with different s
        if time.time() < deadline - 5.0:
            try:
                opt_pts, opt_cr = _lbfgs_riesz(best_pts, s=20.0, maxiter=100, deadline=deadline)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
    
    # Ensure return is valid
    best_pts = _normalize(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
