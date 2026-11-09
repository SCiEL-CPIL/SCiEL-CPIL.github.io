"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets.
    
    For each triangular facet, the covering radius is the angular distance
    from the facet center (normalized centroid) to the farthest vertex.
    This is the circumradius of the spherical triangle.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            pts = points[simplex]
            # The circumcenter of the triangle on the sphere is along the 
            # normalized centroid direction (for acute triangles)
            c = np.mean(pts, axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            
            # The covering radius for this facet is the max angular distance
            # from c to any of the three vertices
            for p in pts:
                cos_dist = np.dot(c, p)
                cos_dist = np.clip(cos_dist, -1.0, 1.0)
                ang = np.arccos(cos_dist)
                max_r = max(max_r, ang)
        return max_r
    except Exception:
        return np.pi

def _covering_radius_surrogate(points, beta=30.0):
    """
    Smooth surrogate for covering radius using soft-min over facets.
    
    For each point i, compute the angular distance to the nearest facet center
    (from Delaunay triangulation). The covering radius is the max over all
    points of this distance. We use a soft-max to make it differentiable.
    
    Returns (objective_value, gradient) where objective is to be minimized.
    """
    n = len(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use pairwise distances
        dots = points @ points.T
        np.fill_diagonal(dots, -1.0)
        min_dots = np.min(dots, axis=1)
        shifted = min_dots - np.max(min_dots)
        exp_vals = np.exp(beta * shifted)
        log_sum_exp = np.log(np.sum(exp_vals)) + np.max(min_dots) / beta
        ang = np.arccos(np.clip(-log_sum_exp, -1, 1))
        return ang, np.zeros_like(points)
    
    # Compute facet centers (normalized centroids)
    facet_centers = []
    for simplex in hull.simplices:
        c = np.mean(points[simplex], axis=0)
        cn = np.linalg.norm(c)
        if cn > 1e-10:
            facet_centers.append(c / cn)
        else:
            facet_centers.append(np.zeros(3))
    facet_centers = np.array(facet_centers)
    n_facets = len(facet_centers)
    
    if n_facets == 0:
        return np.pi, np.zeros_like(points)
    
    # For each point i, compute angular distance to each facet center
    # ang_dist(i, f) = arccos(dot(points[i], facet_centers[f]))
    dots = points @ facet_centers.T  # shape (n, n_facets)
    dots = np.clip(dots, -1.0, 1.0)
    ang_dists = np.arccos(dots)  # shape (n, n_facets)
    
    # For each point, the relevant distance is the MIN over facets 
    # (the point is covered by the nearest facet center in terms of angular distance)
    # Actually, we want: for each point, what's the max angular distance to any 
    # point on the sphere in its Voronoi cell. This equals the max over facets 
    # adjacent to that point of the circumradius.
    
    # Simpler approach: the covering radius is the max over all facets of their circumradius.
    # Each facet's circumradius = max angular distance from facet center to its vertices.
    
    facet_radii = np.zeros(n_facets)
    for idx, simplex in enumerate(hull.simplices):
        fc = facet_centers[idx]
        max_r = 0.0
        for p in points[simplex]:
            d = np.dot(fc, p)
            d = np.clip(d, -1.0, 1.0)
            max_r = max(max_r, np.arccos(d))
        facet_radii[idx] = max_r
    
    # The covering radius is the max over facet_radii
    # Use soft-max to make it differentiable
    max_r = np.max(facet_radii)
    shifted = facet_radii - max_r
    exp_vals = np.exp(beta * shifted)
    exp_vals = np.clip(exp_vals, 1e-300, 1e100)
    log_sum_exp = np.log(np.sum(exp_vals)) + max_r / beta
    
    objective = log_sum_exp
    
    # Gradient: d objective / d points[i]
    # For each facet f, d facet_radii[f] / d points[i] is nonzero only if i is in facet f
    # d arccos(dot(fc, p)) / d p = -fc / sqrt(1 - dot(fc,p)^2)
    # But fc also depends on points, so we need chain rule
    
    # Simplified gradient: for each facet, the gradient of its circumradius
    grad = np.zeros_like(points)
    for idx, simplex in enumerate(hull.simplices):
        fc = facet_centers[idx]
        w = np.exp(beta * (facet_radii[idx] - max_r)) / np.sum(exp_vals)
        if w < 1e-10:
            continue
        for pi in simplex:
            p = points[pi]
            d = np.dot(fc, p)
            d = np.clip(d, -1.0, 1.0)
            s = np.sqrt(max(1e-10, 1.0 - d * d))
            # d arccos(d) / d p = -fc / s
            grad[pi] += w * beta * (-fc / s)
    
    return objective, grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _fibonacci_sphere(n, rng=None):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_covering(points, max_iters=200, lr_init=0.02, beta=30.0, tol=1e-9, time_limit=120.0):
    """
    Gradient descent on the smooth covering radius surrogate.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_obj = _covering_radius_surrogate(pts, beta)[0]
    
    lr = lr_init
    momentum = np.zeros_like(pts)
    mu = 0.85  # Momentum coefficient
    
    start = time.time()
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        obj, grad = _covering_radius_surrogate(pts, beta)
        
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update momentum
        momentum = mu * momentum - lr * tangent_grad
        
        # Take step
        new_pts = pts + momentum
        
        # Project to sphere
        new_pts = _project_to_sphere(new_pts)
        
        # Check if objective improved
        new_obj = _covering_radius_surrogate(new_pts, beta)[0]
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_pts = new_pts.copy()
            pts = new_pts
            # Slowly decay learning rate
            if it % 30 == 0 and lr > 0.001:
                lr *= 0.9
        else:
            # Reduce step size
            lr *= 0.7
            momentum = mu * momentum
        
        # Convergence check
        if it > 30 and abs(obj - best_obj) < tol:
            break
    
    return best_pts, best_obj

def _optimize_combined(points, time_limit=120.0):
    """
    Combined optimization: use covering radius surrogate for gradient-based search,
    then polish with exact covering radius evaluation.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Gradient-based optimization on smooth surrogate
    start = time.time()
    opt_pts, _ = _optimize_covering(pts, max_iters=300, beta=30.0, time_limit=time_limit * 0.5)
    cr1 = _covering_radius(opt_pts)
    if cr1 < best_cr:
        best_cr = cr1
        best_pts = opt_pts.copy()
    
    # Phase 2: Fine-tune with smaller learning rate
    remaining = time_limit - (time.time() - start)
    if remaining > 10:
        fine_pts, _ = _optimize_covering(best_pts, max_iters=200, lr_init=0.005, beta=50.0, time_limit=remaining * 0.7)
        cr2 = _covering_radius(fine_pts)
        if cr2 < best_cr:
            best_cr = cr2
            best_pts = fine_pts.copy()
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_pts = None
    best_cr = np.pi
    
    rng = np.random.default_rng(42)
    
    # Multiple initializations
    initializations = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib)
    
    # Random
    pts_rand = rng.standard_normal((N, 3))
    pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
    initializations.append(pts_rand)
    
    # Perturbed Fibonacci
    pts_pert = pts_fib + 0.1 * rng.standard_normal((N, 3))
    pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
    initializations.append(pts_pert)
    
    # More random restarts with different seeds
    for seed in [1, 7, 13, 99]:
        rng_s = np.random.default_rng(seed)
        pts_r = rng_s.standard_normal((N, 3))
        pts_r /= np.linalg.norm(pts_r, axis=1, keepdims=True)
        initializations.append(pts_r)
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        if remaining < 20:
            break
        
        # Run combined optimization with time budget
        opt_pts, cr = _optimize_combined(init_pts, time_limit=remaining * 0.7)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts.copy()
        
        # If we have time, do additional fine-tuning on the best so far
        if time.time() < deadline - 25 and best_pts is not None:
            remaining2 = deadline - time.time()
            if remaining2 > 15:
                fine_pts, fine_cr = _optimize_combined(best_pts, time_limit=remaining2 * 0.6)
                if fine_cr < best_cr:
                    best_cr = fine_cr
                    best_pts = fine_pts.copy()
    
    if best_pts is None:
        best_pts = initializations[0]
    
    return best_pts
# EVOLVE-BLOCK-END
