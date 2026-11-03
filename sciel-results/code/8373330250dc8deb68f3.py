"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 10000
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


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lloyd_iteration(points, max_iters=2000, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
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


def _soft_max_objective_and_grad(x, temperature=0.1):
    """
    Compute soft-max approximation of covering radius and its gradient.
    
    The covering radius is the max over all facets of the circumradius.
    We approximate max(r_i) with log-sum-exp: (1/T) * log(sum(exp(T * r_i)))
    
    x is a flat array of shape (3*N) representing N points on the sphere.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use KD-tree based approximation
        tree = cKDTree(pts)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        max_d = np.max(dists)
        max_r = 2.0 * np.arcsin(np.clip(max_d / 2.0, 0, 1))
        # Gradient approximation: push points away from nearest sample
        grad = np.zeros_like(pts)
        for i in range(N):
            nearest_idx = tree.query(samples, k=1, workers=-1)[1][0]
            # This is not quite right for gradient, use a simple heuristic
            break
        return max_r, grad.flatten()
    
    # Compute circumradius for each facet
    radii = []
    centers = []
    for simplex in hull.simplices:
        pts_s = pts[simplex]
        if len(pts_s) != 3:
            continue
        a, b, c = pts_s[0], pts_s[1], pts_s[2]
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
        centers.append((v, simplex))
    
    if not radii:
        return 0.0, np.zeros_like(x)
    
    radii = np.array(radii)
    
    # Soft-max approximation
    T = temperature
    scaled = T * radii
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    obj = log_sum_exp / T
    
    # Gradient of soft-max: d/dx (1/T * log(sum(exp(T*r_i)))) = sum(exp(T*r_i) * T * dr_i/dx) / sum(exp(T*r_i))
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # Gradient of circumradius w.r.t. each point
    # For a facet with vertices a, b, c and circumcenter v:
    # r = arccos(v · a), so dr/da = -v / sqrt(1 - (v·a)^2)
    # Projected onto tangent plane: dr/da_tangent = dr/da - (dr/da · a) * a
    grad_pts = np.zeros_like(pts)
    
    for (v, simplex), w in zip(centers, weights):
        for idx in simplex:
            p = pts[idx]
            dot_vp = np.dot(v, p)
            sin_r = np.sqrt(np.maximum(0, 1.0 - dot_vp * dot_vp))
            if sin_r < 1e-12:
                continue
            # dr/dp = -v / sin_r
            dr_dp = -v / sin_r
            # Project onto tangent plane
            dr_dp_tangent = dr_dp - np.dot(dr_dp, p) * p
            grad_pts[idx] += w * dr_dp_tangent
    
    return obj, grad_pts.flatten()


def _lbfgs_refine(points, time_limit=None, start_time=None, temperature=0.1):
    """Refine points using L-BFGS-B with soft-max objective."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def obj_and_grad(x):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                return _covering_radius(x.reshape(N, 3)), np.zeros_like(x)
        val, grad = _soft_max_objective_and_grad(x, temperature)
        return val, grad
    
    try:
        result = minimize(
            lambda x: _soft_max_objective_and_grad(x, temperature)[0],
            x0,
            jac=lambda x: _soft_max_objective_and_grad(x, temperature)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success and result.fun < best_cr:
            new_pts = result.x.reshape(N, 3)
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts
    except Exception:
        pass
    
    return best_points, best_cr


def _perturb_and_refine(points, n_perturbs=10, time_limit=None, start_time=None):
    """Perturb random points and refine with local optimization."""
    points = np.array(points, dtype=float)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(42)
    
    for _ in range(n_perturbs):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * 0.05
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        refined, cr1 = _lloyd_iteration(perturbed, max_iters=500, time_limit=5, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        refined, cr2 = _lbfgs_refine(refined, time_limit=5, start_time=start_time, temperature=0.1)
        
        if cr2 < best_cr:
            best_cr = cr2
            best_points = refined.copy()
    
    return best_points, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_points(N)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Lloyd's algorithm from Fibonacci start
    remaining = deadline - time.time()
    if remaining > 20:
        points, cr = _lloyd_iteration(points, max_iters=3000, time_limit=remaining * 0.25, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 2: L-BFGS refinement with soft-max objective
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _lbfgs_refine(best_points, time_limit=remaining * 0.3, start_time=start_time, temperature=0.1)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Perturb and refine to escape local minima
    remaining = deadline - time.time()
    if remaining > 10:
        points, cr = _perturb_and_refine(best_points, n_perturbs=8, time_limit=remaining * 0.4, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 4: Final Lloyd + L-BFGS polish
    remaining = deadline - time.time()
    if remaining > 5:
        points, cr = _lloyd_iteration(best_points, max_iters=2000, time_limit=remaining * 0.3, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        remaining = deadline - time.time()
        if remaining > 3:
            points, cr = _lbfgs_refine(best_points, time_limit=remaining * 0.7, start_time=start_time, temperature=0.05)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
