import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_fast(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
    
    if len(hull.simplices) == 0:
        return np.pi, None
    
    max_r = 0.0
    worst_c = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute normal to the facet
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            r = np.pi
            cc = np.zeros(3)
        else:
            n = n / nn
            best_r = np.pi
            best_c = n
            for s in (1.0, -1.0):
                p = s * n
                da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
                db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
                dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
                if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
                    if da < best_r:
                        best_r = da
                        best_c = p
            r = best_r
            cc = best_c
        
        if r > max_r:
            max_r = r
            worst_c = cc
    
    return max_r, worst_c

def _smooth_covering_grad(pts, beta=200.0):
    """
    Compute the smooth covering radius and its gradient with respect to points.
    Uses log-sum-exp for the soft maximum of the minimum distances.
    """
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances
    dots = samples @ pts.T  # (n_samples, N)
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)  # (n_samples, N)
    
    min_idx = np.argmin(angles, axis=1)  # (n_samples,)
    min_angles = angles[np.arange(n_samples), min_idx]  # (n_samples,)
    
    # Soft max weights
    max_angle = np.max(min_angles)
    log_terms = beta * (min_angles - max_angle)
    log_terms = np.where(log_terms > -50, log_terms, -50)
    exp_terms = np.exp(log_terms)
    weights = exp_terms / np.sum(exp_terms)  # (n_samples,)
    
    smooth_val = max_angle - np.log(np.sum(exp_terms)) / beta
    
    # Gradient computation
    grad = np.zeros_like(pts)
    for i in range(n_samples):
        j = min_idx[i]
        w = weights[i]
        a = min_angles[i]
        sin_a = np.sin(a)
        if sin_a < 1e-10:
            continue
        # Gradient of angle w.r.t. pts[j]
        d_angle_d_pt = -samples[i] / sin_a
        grad[j] += w * d_angle_d_pt
    
    return smooth_val, -grad

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    return _to_sphere(pts)

def _repulsion_start(n, seed=None):
    """Generate points using repulsion forces to get a well-distributed start."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    pts = _to_sphere(pts)
    
    # Simple repulsion simulation
    for _ in range(100):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-6:
                    continue
                # Repulsion force proportional to 1/d^2
                f = (pts[i] - pts[j]) / (d**2)
                forces[i] += f
        pts = _to_sphere(pts + 0.01 * forces)
    
    return pts

def _lloyd_relax(pts, max_iter=30, deadline=None):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    """
    pts = _to_sphere(pts.copy())
    current_cr, _ = _covering_radius_fast(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute all facet circumradii and identify targets
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        targets = np.zeros_like(pts)
        weights = np.zeros(len(pts))
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                r = np.pi
                cc = np.zeros(3)
            else:
                n = n / nn
                best_r = np.pi
                best_c = n
                for s in (1.0, -1.0):
                    p = s * n
                    da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
                    db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
                    dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
                    if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
                        if da < best_r:
                            best_r = da
                            best_c = p
                r = best_r
                cc = best_c
            
            w = r * r
            for idx in simplex:
                targets[idx] += w * cc
                weights[idx] += w
        
        improved = False
        for idx in range(len(pts)):
            if weights[idx] > 1e-12:
                target = targets[idx] / np.linalg.norm(targets[idx])
                new_pt = 0.5 * pts[idx] + 0.5 * target
                new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
                pts[idx] = new_pt
                improved = True
        
        if not improved:
            break
        
        new_cr, _ = _covering_radius_fast(pts)
        if new_cr < current_cr - 1e-10:
            current_cr = new_cr
        else:
            break
    
    return pts, current_cr

def _gradient_descent(pts, max_iter=50, deadline=None):
    """
    Multi-stage beta cascade gradient descent on the smooth covering radius approximation.
    """
    pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius_fast(pts)
    best_pts = pts.copy()
    best_val = current_val
    
    # Multi-stage beta cascade
    beta_stages = [
        (20.0, 10, 1e-8, 1e-6),
        (50.0, 15, 1e-10, 1e-8),
        (100.0, 20, 1e-12, 1e-10),
        (200.0, 25, 1e-14, 1e-12),
        (400.0, 30, 1e-16, 1e-14),
    ]
    
    for beta, stage_maxiter, ftol, gtol in beta_stages:
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        x0 = pts.flatten()
        
        def obj(x, b=beta):
            p = _to_sphere(x.reshape(N, 3))
            val, _ = _smooth_covering_grad(p, beta=b)
            return val
        
        def obj_grad(x, b=beta):
            p = _to_sphere(x.reshape(N, 3))
            val, grad = _smooth_covering_grad(p, beta=b)
            # Project gradient to tangent space of sphere
            grad_proj = np.zeros_like(x)
            for j in range(N):
                pj = p[j]
                gj = grad[j]
                gj_tan = gj - np.dot(gj, pj) * pj
                grad_proj[j * 3:(j + 1) * 3] = gj_tan
            return grad_proj.flatten()
        
        try:
            res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                          options={'maxiter': stage_maxiter, 'ftol': ftol, 'gtol': gtol})
            if res.success:
                new_pts = _to_sphere(res.x.reshape(N, 3))
                new_val, _ = _covering_radius_fast(new_pts)
                if new_val < best_val - 1e-10:
                    pts = new_pts
                    best_pts = new_pts.copy()
                    best_val = new_val
        except Exception:
            pass
    
    return best_pts, best_val

def _local_refine(pts, max_iter=50, deadline=None):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c = _covering_radius_fast(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Try different step sizes
        improved = False
        for step in [0.05, 0.02, 0.01, 0.005, 0.001]:
            new_pt = pts[k] + step * worst_c
            new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
            
            cand = pts.copy()
            cand[k] = new_pt
            cr, wc = _covering_radius_fast(cand)
            
            if cr < best_cr - 1e-9:
                pts = cand
                best_cr = cr
                worst_c = wc
                improved = True
                break
        
        if not improved:
            break
    
    return pts, best_cr

def _high_precision_polish(pts, deadline=None):
    """
    Final high-precision polish pass using very tight L-BFGS tolerances
    with high beta for the smooth surrogate.
    """
    pts = _to_sphere(pts.copy())
    best_pts = pts.copy()
    best_val, _ = _covering_radius_fast(pts)
    
    # Very high beta for precise local optimization
    beta = 500.0
    ftol = 1e-16
    gtol = 1e-14
    
    x0 = pts.flatten()
    
    def obj(x):
        p = _to_sphere(x.reshape(N, 3))
        val, _ = _smooth_covering_grad(p, beta=beta)
        return val
    
    def obj_grad(x):
        p = _to_sphere(x.reshape(N, 3))
        val, grad = _smooth_covering_grad(p, beta=beta)
        grad_proj = np.zeros_like(x)
        for j in range(N):
            pj = p[j]
            gj = grad[j]
            gj_tan = gj - np.dot(gj, pj) * pj
            grad_proj[j * 3:(j + 1) * 3] = gj_tan
        return grad_proj.flatten()
    
    try:
        res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                      options={'maxiter': 50, 'ftol': ftol, 'gtol': gtol})
        if res.success:
            new_pts = _to_sphere(res.x.reshape(N, 3))
            new_val, _ = _covering_radius_fast(new_pts)
            if new_val < best_val:
                best_pts = new_pts.copy()
                best_val = new_val
    except Exception:
        pass
    
    # Additional local refinement with very small steps
    for it in range(50):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        cr, wc = _covering_radius_fast(best_pts)
        if wc is None:
            break
        
        d = np.sum((best_pts - wc[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        for step in [0.005, 0.002, 0.001, 0.0005]:
            new_pt = best_pts[k] + step * wc
            new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
            cand = best_pts.copy()
            cand[k] = new_pt
            cr2, _ = _covering_radius_fast(cand)
            if cr2 < cr - 1e-10:
                best_pts = cand
                break
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    _deadline = time.time() + 140.0  # safety margin vs 150s kill
    
    np.random.seed(42)
    
    best_pts = None
    best_val = np.pi
    
    # Multiple starting configurations
    starts = [_fibonacci_spiral(N)]
    for seed in [1, 2, 3, 4]:
        starts.append(_random_points(N, seed=seed))
    for seed in [5, 6]:
        starts.append(_repulsion_start(N, seed=seed))
    
    for st in starts:
        if time.time() > _deadline - 25.0:
            break
        
        # Apply Lloyd's relaxation
        lloyd_pts, lloyd_val = _lloyd_relax(st, max_iter=30, deadline=_deadline)
        
        # Apply multi-stage beta cascade gradient descent
        gd_pts, gd_val = _gradient_descent(lloyd_pts, max_iter=50, deadline=_deadline)
        
        # Apply local refinement
        ref_pts, ref_val = _local_refine(gd_pts, max_iter=50, deadline=_deadline)
        
        if ref_val < best_val:
            best_val = ref_val
            best_pts = ref_pts.copy()
    
    # Final local refinement on the best result
    if best_pts is not None and time.time() < _deadline - 10.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=50, deadline=_deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    # High-precision polish pass
    if best_pts is not None and time.time() < _deadline - 5.0:
        polish_pts, polish_val = _high_precision_polish(best_pts, deadline=_deadline)
        if polish_val < best_val:
            best_val = polish_val
            best_pts = polish_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _to_sphere(best_pts)

# EVOLVE-BLOCK-END
