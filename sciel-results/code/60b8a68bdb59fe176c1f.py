"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = _normalize(np.asarray(points, dtype=float))
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r


def _facet_circumradii(points):
    """Compute circumradii and facet data for all triangular facets."""
    points = _normalize(np.asarray(points, dtype=float))
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    radii = []
    simplices = []
    centers = []
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        radii.append(r)
        simplices.append(simplex)
        centers.append(v)
    
    return np.array(radii) if radii else None, simplices, centers


def _smooth_objective(x, beta=50.0):
    """Smooth surrogate: log-sum-exp of facet circumradii."""
    points = _normalize(x.reshape(-1, 3))
    radii, simplices, centers = _facet_circumradii(points)
    if radii is None or len(radii) == 0:
        return 1.0, np.zeros_like(x)
    
    # Log-sum-exp for numerical stability
    max_r = np.max(radii)
    log_sum = np.log(np.sum(np.exp(radii - max_r))) + max_r
    
    # Gradient: softmax weights times d(r_i)/dx
    weights = np.exp(radii - max_r) / np.sum(np.exp(radii - max_r))
    
    grad = np.zeros_like(x)
    for i, (w, simp, center) in enumerate(zip(weights, simplices, centers)):
        for idx in simp:
            p = points[idx]
            d = center - p
            d_tangent = d - np.dot(d, p) * p
            grad[idx * 3: idx * 3 + 3] += w * d_tangent
    
    return log_sum, grad


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]


def _lloyd_iteration(points, max_iters=200, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm (Voronoi relaxation)."""
    points = _normalize(np.array(points, dtype=float))
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _lbfgs_smooth_minimize(points, time_limit=None, start_time=None, beta=50.0):
    """L-BFGS optimization on smooth log-sum-exp surrogate."""
    points = _normalize(np.array(points, dtype=float))
    x0 = points.flatten().copy()
    best_x = x0.copy()
    best_val = _covering_radius(points)
    
    def objective(x):
        val, _ = _smooth_objective(x, beta=beta)
        return val
    
    def gradient(x):
        _, grad = _smooth_objective(x, beta=beta)
        return grad
    
    try:
        result = minimize(
            objective, x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
        )
        if result.x is not None:
            new_points = _normalize(result.x.reshape(-1, 3))
            new_cr = _covering_radius(new_points)
            if new_cr < best_val:
                best_val = new_cr
                best_x = result.x.copy()
    except Exception:
        pass
    
    return _normalize(best_x.reshape(-1, 3)), best_val


def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Lloyd + L-BFGS smooth optimization."""
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        lloyd_time = min(remaining * 0.25, 8)
    else:
        lloyd_time = 8
    
    points, cr1 = _lloyd_iteration(points, max_iters=150, time_limit=lloyd_time, start_time=start_time)
    
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, cr1
        lbfgs_time = min(remaining * 0.7, 40)
    else:
        lbfgs_time = 40
    
    points, cr2 = _lbfgs_smooth_minimize(points, time_limit=lbfgs_time, start_time=start_time, beta=50.0)
    
    return points, cr2


def _polish(points, time_limit=None, start_time=None):
    """High-precision polish with tighter tolerances."""
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1:
            return points, _covering_radius(points)
    
    # Use higher beta for sharper approximation to max
    points, cr = _lbfgs_smooth_minimize(points, time_limit=time_limit, start_time=start_time, beta=200.0)
    
    # Final Lloyd pass with tight tolerance
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 2:
            points, cr2 = _lloyd_iteration(points, max_iters=50, tol=1e-12, time_limit=min(remaining * 0.5, 5), start_time=start_time)
            if cr2 < cr:
                cr = cr2
    
    return points, cr


def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration with Lloyd + L-BFGS
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        time_per_config = min(remaining / (len(initial_configs) - i), 30)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Polish the best configuration
    remaining = deadline - time.time()
    if remaining > 10:
        polish_time = min(remaining * 0.6, 30)
        polished, pol_cr = _polish(best_points, time_limit=polish_time, start_time=start_time)
        if pol_cr < best_cr:
            best_cr = pol_cr
            best_points = polished.copy()
    
    # Phase 3: One more L-BFGS pass from the polished result
    remaining = deadline - time.time()
    if remaining > 5:
        final_time = min(remaining * 0.7, 20)
        final_points, final_cr = _lbfgs_smooth_minimize(best_points, time_limit=final_time, start_time=start_time, beta=100.0)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
