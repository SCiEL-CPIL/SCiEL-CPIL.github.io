"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin below 120s

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius in degrees via convex-hull facet circumcenters.
    For each triangular facet, the circumcenter (on the sphere) is the farthest
    point in that Voronoi cell. The covering radius is the max angular distance
    from any facet circumcenter to its vertices.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: sample-based estimation
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(pts)
        n_samples = 50000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))) * 180.0 / np.pi

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(np.degrees(max_r))

def _fast_facets(pts):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
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
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius (log-sum-exp over facet radii) and its
    gradient with respect to the input points (tangent space).
    """
    points = _normalize(points)
    facets = _fast_facets(points)
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
    """Refine points using L-BFGS-B with multi-stage beta cascade."""
    from scipy.optimize import minimize

    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    # Multi-stage beta cascade: progressively tighter
    beta_stages = [(15, 1e-10, 1e-8, 200), (50, 1e-12, 1e-10, 300), (150, 1e-14, 1e-12, 300), (300, 1e-15, 1e-13, 400), (500, 1e-16, 1e-14, 500)]
    
    for beta_s, ftol, gtol, maxiter in beta_stages:
        if time.time() - start_time > max_time - 2.0:
            break
        
        def objective_and_grad(x_flat, b=beta_s):
            x = x_flat.reshape(-1, 3)
            x = _normalize(x)
            val, grad = _smooth_covering_radius_grad(x, beta=b)
            grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
            return val, grad_tan.flatten()

        x0 = best.flatten()
        try:
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 3}
            )
            x_opt = _normalize(result.x.reshape(-1, 3))
            cr_opt = _covering_radius(x_opt)
            if cr_opt < best_cr:
                best = x_opt.copy()
                best_cr = cr_opt
        except Exception:
            pass

    return best, best_cr

def _worst_facet_shrinkage(points, max_iters=500, start_time=None, time_limit=None):
    """
    Deterministic iterative method: identify the convex hull facet with the largest
    spherical circumradius and move its three vertices along the great circle toward
    the facet's circumcenter. This directly shrinks the worst empty cap.
    """
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)
    
    if start_time is None:
        start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        facets = _fast_facets(points)
        if not facets:
            break
        
        # Find the worst facet
        worst_idx = np.argmax([f[0] for f in facets])
        worst_r, worst_center, worst_simplex = facets[worst_idx]
        
        if worst_r < 1e-10:
            break
        
        # Move the three vertices of the worst facet toward the circumcenter
        # using spherical interpolation (great circle)
        step_size = 0.3  # fraction of the way to move
        new_points = points.copy()
        for idx in worst_simplex:
            p = points[idx]
            c = worst_center
            # Spherical interpolation: move p toward c by step_size * angular distance
            cos_angle = np.clip(np.dot(p, c), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < 1e-10:
                continue
            # Interpolate along great circle
            t = step_size
            sin_angle = np.sin(angle)
            if sin_angle < 1e-10:
                continue
            new_p = (np.sin((1-t)*angle) * p + np.sin(t*angle) * c) / sin_angle
            new_points[idx] = new_p
        
        new_points = _normalize(new_points)
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr - 1e-12:
            points = new_points
            best = points.copy()
            best_cr = new_cr
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 0.01:
                break

    return best, best_cr

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """Lloyd-style: each point moves to the average circumcenter of adjacent facets."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        facets = _fast_facets(points)
        if not facets:
            break
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended = _normalize(blended)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended = _normalize(blended)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break

    return best, best_cr

def _fibonacci_points(n=N):
    """Generate points using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    """Generate random points on sphere."""
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)

def _icosahedron_based_points():
    """Generate points based on icosahedron symmetry, then add more."""
    # Start with icosahedron vertices (12 points)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = _normalize(ico)
    
    # Add 38 more points using Fibonacci spiral, then optimize
    fib = _fibonacci_points(N)
    # Blend: use icosahedron as base, add perturbed Fibonacci points
    # Actually, just use Fibonacci as it's a good uniform distribution
    return fib

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Golden-angle spiral
    starts.append(_fibonacci_points())
    
    # 2. Fewer random starts (concentrate compute)
    for seed in [1, 7, 42]:
        starts.append(_random_points(seed))
    
    # 3. Perturbed golden spiral (small)
    np.random.seed(7)
    base = starts[0].copy()
    noise = np.random.randn(N, 3) * 0.05
    starts.append(_normalize(base + noise))
    
    # 4. Perturbed golden spiral (medium)
    np.random.seed(13)
    noise = np.random.randn(N, 3) * 0.2
    starts.append(_normalize(base + noise))
    
    # 5. Perturbed golden spiral (larger)
    np.random.seed(99)
    noise = np.random.randn(N, 3) * 0.5
    starts.append(_normalize(base + noise))
    
    # Optimize from each start, keep the best
    best_pts = starts[0]
    best_val = _covering_radius(best_pts)
    
    # Allocate time per start
    n_starts = len(starts)
    elapsed = time.time() - START_TIME
    time_per_start = max(3.0, (TIME_LIMIT - elapsed - 15.0) / n_starts)
    
    for idx, start in enumerate(starts):
        remaining = TIME_LIMIT - (time.time() - START_TIME) - 10.0
        if remaining < 5.0:
            break
        
        max_time_for_this = min(time_per_start, remaining - 5.0)
        if max_time_for_this < 3.0:
            break
        
        try:
            # Phase 1: Worst facet shrinkage (fast, deterministic)
            pts, val = _worst_facet_shrinkage(start, max_iters=200, 
                                               start_time=START_TIME,
                                               time_limit=time.time() - START_TIME + max_time_for_this * 0.3)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            # Phase 2: Lloyd iteration
            pts, val = _spherical_lloyd(pts, max_iters=100,
                                         start_time=START_TIME,
                                         time_limit=time.time() - START_TIME + max_time_for_this * 0.6)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            # Phase 3: L-BFGS refinement
            remaining_for_lbfgs = max(2.0, max_time_for_this * 0.4)
            pts, val = _lbfgs_refine(pts, max_time=remaining_for_lbfgs,
                                      start_time=START_TIME)
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time allows
    remaining = TIME_LIMIT - (time.time() - START_TIME) - 3.0
    if remaining > 15.0:
        try:
            # More aggressive refinement
            pts, val = _worst_facet_shrinkage(best_pts, max_iters=300,
                                               start_time=START_TIME,
                                               time_limit=time.time() - START_TIME + remaining * 0.3)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            pts, val = _spherical_lloyd(pts, max_iters=200,
                                         start_time=START_TIME,
                                         time_limit=time.time() - START_TIME + remaining * 0.5)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            remaining_for_lbfgs = max(5.0, remaining * 0.4)
            pts, val = _lbfgs_refine(pts, max_time=remaining_for_lbfgs,
                                      start_time=START_TIME)
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            pass
    
    # Final safety check
    remaining = TIME_LIMIT - (time.time() - START_TIME)
    if remaining < 5.0:
        # Return immediately
        return _normalize(best_pts)
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
