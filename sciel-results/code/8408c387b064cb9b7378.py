"""Optimized covering of sphere with 50 points using smooth surrogate + L-BFGS-B."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import sys

N = 50
_TIME_LIMIT = 145.0
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
            a, b, c = tri
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            t = np.dot(a, n)
            circumcenter_3d = t * n
            cc_norm = np.linalg.norm(circumcenter_3d)
            if cc_norm < 1e-15:
                continue
            circumcenter_sphere = circumcenter_3d / cc_norm
            for v in [a, b, c]:
                cos_ang = np.clip(np.dot(circumcenter_sphere, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                if ang > max_cr:
                    max_cr = ang
        return np.degrees(max_cr)
    except Exception:
        return 180.0


def _fibonacci_points(jitter=0.0, seed=None):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = rng.randn(N, 3) * jitter
        pts = pts + noise
        pts = _normalize(pts)
    return pts


def _random_points(seed=None):
    """Random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    return _normalize(pts)


def _spherical_lloyd(pts, n_iters=30, tol=1e-8):
    """Spherical Lloyd's algorithm: move each point to centroid of its Voronoi cell."""
    pts = _normalize(pts.copy())
    for it in range(n_iters):
        if _time_remaining() < 10.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        new_pts = pts.copy()
        for i in range(N):
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
        
        diff = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = _normalize(new_pts)
        if diff < tol:
            break
    
    return pts


def _soft_max_covering_radius_and_grad(x, beta, sample_pts):
    """
    Compute soft max covering radius and its gradient.
    x: (N, 3) array of points on unit sphere
    beta: temperature parameter
    sample_pts: (M, 3) array of sample points on sphere
    
    Returns: (soft_max_value, gradient) where gradient is (N, 3)
    """
    pts = _normalize(x.reshape(N, 3))
    M = sample_pts.shape[0]
    
    # Compute dot products: (M, N)
    dots = sample_pts @ pts.T
    cos_dists = np.clip(dots, -1.0 + 1e-10, 1.0 - 1e-10)
    
    # Angular distances: (M, N)
    ang_dists = np.arccos(cos_dists)
    
    # For each sample point, find the minimum angular distance
    min_ang = ang_dists.min(axis=1)  # (M,)
    
    # Soft max over the minimum distances
    scaled = beta * min_ang
    max_scaled = scaled.max()
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / exp_scaled.sum()  # (M,)
    
    soft_max_val = max_scaled / beta + np.log(exp_scaled.sum()) / beta
    
    # Gradient computation
    # d(soft_max)/d(x_i) = sum_m weights[m] * d(min_ang[m])/d(x_i)
    # min_ang[m] = min_k ang_dists[m, k]
    # For the k that achieves the min: d(ang_dists[m,k])/d(x_k)
    
    # Find which point achieves the minimum for each sample
    min_idx = ang_dists.argmin(axis=1)  # (M,)
    
    grad = np.zeros((N, 3))
    
    # For each sample, compute gradient contribution
    # ang = arccos(cos), d(ang)/d(cos) = -1/sqrt(1-cos^2)
    # cos = sample_pt @ x_k, d(cos)/d(x_k) = sample_pt
    # So d(ang)/d(x_k) = -sample_pt / sqrt(1 - cos^2)
    
    for m in range(M):
        k = min_idx[m]
        w = weights[m]
        if w < 1e-12:
            continue
        cos_val = cos_dists[m, k]
        sin_val = np.sqrt(np.clip(1.0 - cos_val * cos_val, 1e-10, 1.0))
        # d(ang)/d(x_k) = -sample_pts[m] / sin_val
        grad[k] += w * (-sample_pts[m] / sin_val)
    
    # Project gradient to tangent space (keep only components orthogonal to each point)
    for i in range(N):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return soft_max_val, grad


def _lbfgs_optimize(pts, beta_start=20.0, beta_end=200.0, n_stages=5, 
                    n_samples=400, max_iter=50, tol=1e-10):
    """
    Multi-stage L-BFGS-B optimization with increasing beta.
    Starts with low beta (smooth) and increases to high beta (sharp).
    """
    pts = _normalize(pts.copy())
    
    # Generate fixed sample points (deterministic for reproducibility)
    rng = np.random.RandomState(12345)
    sample_pts = rng.randn(n_samples, 3)
    sample_pts = _normalize(sample_pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Beta schedule: geometric from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_stages)
    
    for stage, beta in enumerate(betas):
        if _time_remaining() < 15.0:
            break
        
        # Define objective and gradient
        def objective(x):
            val, _ = _soft_max_covering_radius_and_grad(x, beta, sample_pts)
            return val
        
        def gradient(x):
            _, grad = _soft_max_covering_radius_and_grad(x, beta, sample_pts)
            return grad.flatten()
        
        # Flatten pts for L-BFGS-B
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'ftol': tol,
                    'gtol': 1e-6,
                    'maxls': 20,
                }
            )
            
            if result.success or result.fun < objective(x0):
                new_pts = _normalize(result.x.reshape(N, 3))
                cr = _covering_radius(new_pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = new_pts.copy()
                pts = new_pts
            else:
                # If optimization failed, keep current pts
                pass
        except Exception:
            pass
    
    return best_pts, best_cr


def _facet_polish(pts, n_iters=50):
    """Final polish by moving vertices of worst facet toward circumcenter."""
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(n_iters):
        if _time_remaining() < 10.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_cr = 0.0
        worst_facet = None
        worst_cc = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            t = np.dot(a, n)
            cc_3d = t * n
            cc_norm = np.linalg.norm(cc_3d)
            if cc_norm < 1e-15:
                continue
            cc = cc_3d / cc_norm
            cr = 0.0
            for v in [a, b, c]:
                cos_ang = np.clip(np.dot(cc, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                cr = max(cr, ang)
            if cr > max_cr:
                max_cr = cr
                worst_facet = simplex
                worst_cc = cc
        
        if worst_facet is None:
            break
        
        # Move the vertices of the worst facet toward the circumcenter
        step = 0.03 * (1.0 - it / n_iters)
        for idx in worst_facet:
            pts[idx] += step * (worst_cc - pts[idx])
        pts = _normalize(pts)
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts


def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    np.random.seed(42)
    
    best_pts = None
    best_cr = 180.0
    
    # Phase 1: Multiple diverse starts
    n_starts = 6
    for s_idx in range(n_starts):
        if _time_remaining() < 30.0:
            break
        
        # Diverse initial configurations
        if s_idx % 2 == 0:
            start = _fibonacci_points(jitter=0.1, seed=s_idx)
        else:
            start = _random_points(seed=100 + s_idx)
        
        # Stage 1: Lloyd's algorithm for initial improvement
        pts = _spherical_lloyd(start, n_iters=20)
        
        # Stage 2: Multi-stage L-BFGS-B with soft max surrogate
        pts, cr = _lbfgs_optimize(
            pts, 
            beta_start=15.0, 
            beta_end=150.0, 
            n_stages=4,
            n_samples=350,
            max_iter=40
        )
        
        # Stage 3: Facet polish
        pts = _facet_polish(pts, n_iters=30)
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Phase 2: If time remains, do additional refinement on best
    if best_pts is not None and _time_remaining() > 25.0:
        # More Lloyd iterations
        refined = _spherical_lloyd(best_pts, n_iters=15)
        cr = _covering_radius(refined)
        if cr < best_cr:
            best_cr = cr
            best_pts = refined.copy()
        
        # More L-BFGS with higher beta
        if _time_remaining() > 20.0:
            refined, cr2 = _lbfgs_optimize(
                best_pts,
                beta_start=50.0,
                beta_end=300.0,
                n_stages=3,
                n_samples=400,
                max_iter=30
            )
            if cr2 < best_cr:
                best_cr = cr2
                best_pts = refined.copy()
        
        # Final facet polish
        if _time_remaining() > 10.0:
            best_pts = _facet_polish(best_pts, n_iters=20)
            cr = _covering_radius(best_pts)
            if cr < best_cr:
                best_cr = cr
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_points()
    
    best_pts = _normalize(best_pts)
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10):
        best_pts = _fibonacci_points()
    
    # Check not in closed hemisphere
    try:
        hull = ConvexHull(best_pts)
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
            best_pts = _fibonacci_points()
    except Exception:
        best_pts = _fibonacci_points()
    
    return best_pts
