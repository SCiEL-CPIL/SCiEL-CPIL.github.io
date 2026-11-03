"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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

def _smooth_covering(pts, beta=200.0):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    # Sample points on the sphere using a Fibonacci spiral for efficiency
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distance from each sample to nearest point
    # cos(angle) = dot(product)
    dots = samples @ pts.T  # (n_samples, N)
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)  # (n_samples, N)
    
    # For each sample, find the minimum angle (distance to nearest point)
    min_angles = np.min(angles, axis=1)  # (n_samples,)
    
    # Log-sum-exp for smooth maximum
    beta = 200.0
    max_angle = np.max(min_angles)
    # Soft max approximation
    log_terms = beta * (min_angles - max_angle)
    log_terms = np.where(log_terms > -50, log_terms, -50)  # Avoid underflow
    sum_exp = np.sum(np.exp(log_terms))
    smooth_max = max_angle - np.log(sum_exp) / beta
    
    return smooth_max

def _smooth_covering_grad(pts, beta=200.0):
    """
    Compute the smooth covering radius and its gradient with respect to points.
    """
    n_samples = 1000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find nearest point and compute contribution
    dots = samples @ pts.T  # (n_samples, N)
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
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
    # d(smooth_val)/d(pts[j]) = -sum_i weights[i] * d(min_angles[i])/d(pts[j])
    # For sample i, nearest point is min_idx[i]
    # d(angle)/d(pt) = -sin(angle) / cos(angle) * dot(sample, pt) ... 
    # Actually: angle = arccos(cos_angle), d(angle)/d(cos_angle) = -1/sin(angle)
    # cos_angle = dot(sample, pt), d(cos_angle)/d(pt) = sample
    # So d(angle)/d(pt) = -sample / sin(angle)
    
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
    
    # The gradient of smooth_val w.r.t. pts is -grad (since we're minimizing, 
    # but smooth_val = max - log(sum)/beta, and we want to minimize it)
    # Actually smooth_val = max_angle - (1/beta) * log(sum exp(beta*(a_i - max)))
    # d smooth_val / d pts = -sum_i weights[i] * d a_i / d pts
    # where weights[i] = exp(beta*(a_i - max)) / sum_j exp(beta*(a_j - max))
    # So d smooth_val / d pts[j] = -sum_{i: min_idx[i]=j} weights[i] * d a_i / d pts[j]
    # We computed grad[j] = sum_{i: min_idx[i]=j} weights[i] * d a_i / d pts[j]
    # So d smooth_val / d pts = -grad
    
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

def _lloyd_relax(pts, max_iter=60, deadline=None):
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
                new_pt = 0.7 * pts[idx] + 0.3 * target
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

def _gradient_descent(pts, max_iter=100, deadline=None):
    """
    Gradient descent on the smooth covering radius approximation.
    """
    pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius_fast(pts)
    best_pts = pts.copy()
    best_val = current_val
    
    lr = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Use scipy minimize with L-BFGS-B on the smooth objective
        # Flatten points
        x0 = pts.flatten()
        
        def obj(x):
            p = _to_sphere(x.reshape(N, 3))
            val, _ = _smooth_covering_grad(p, beta=100.0)
            return val
        
        def obj_grad(x):
            p = _to_sphere(x.reshape(N, 3))
            val, grad = _smooth_covering_grad(p, beta=100.0)
            # Project gradient to tangent space of sphere
            grad_proj = np.zeros_like(x)
            for j in range(N):
                pj = p[j]
                gj = grad[j]
                # Tangent space: remove component along pj
                gj_tan = gj - np.dot(gj, pj) * pj
                grad_proj[j * 3:(j + 1) * 3] = gj_tan
            return grad_proj.flatten()
        
        try:
            res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                          options={'maxiter': 5, 'gtol': 1e-4})
            if res.success:
                new_pts = _to_sphere(res.x.reshape(N, 3))
                new_val, _ = _covering_radius_fast(new_pts)
                if new_val < best_val - 1e-10:
                    pts = new_pts
                    best_pts = new_pts.copy()
                    best_val = new_val
        except Exception:
            pass
        
        # Simple gradient step as fallback
        val, grad = _smooth_covering_grad(pts, beta=100.0)
        # Project to tangent space
        grad_proj = np.zeros_like(pts)
        for j in range(N):
            pj = pts[j]
            gj = grad[j]
            gj_tan = gj - np.dot(gj, pj) * pj
            grad_proj[j] = gj_tan
        
        new_pts = pts - lr * grad_proj
        new_pts = _to_sphere(new_pts)
        new_val, _ = _covering_radius_fast(new_pts)
        
        if new_val < best_val - 1e-10:
            pts = new_pts
            best_pts = new_pts.copy()
            best_val = new_val
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_pts, best_val

def _local_refine(pts, max_iter=40, step0=0.03, deadline=None):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c = _covering_radius_fast(pts)
    step = step0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Move that point toward the circumcenter
        new_pt = pts[k] + step * worst_c
        new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
        
        cand = pts.copy()
        cand[k] = new_pt
        cr, wc = _covering_radius_fast(cand)
        
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            worst_c = wc
            step = min(step * 1.1, 0.15)
        else:
            step *= 0.8
            if step < 1e-5:
                break
    
    return pts, best_cr

def _simulated_annealing(pts, max_iter=150, T0=0.2, T_end=0.005, deadline=None):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.08 * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val, _ = _covering_radius_fast(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
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
    
    for st in starts:
        if time.time() > _deadline - 25.0:
            break
        
        # Apply Lloyd's relaxation
        lloyd_pts, lloyd_val = _lloyd_relax(st, max_iter=50, deadline=_deadline)
        
        # Apply gradient descent on smooth objective
        gd_pts, gd_val = _gradient_descent(lloyd_pts, max_iter=80, deadline=_deadline)
        
        # Apply local refinement
        ref_pts, ref_val = _local_refine(gd_pts, max_iter=30, step0=0.025, deadline=_deadline)
        
        # Apply simulated annealing for final polish
        sa_pts, sa_val = _simulated_annealing(ref_pts, max_iter=100, T0=0.15, T_end=0.005, deadline=_deadline)
        
        if sa_val < best_val:
            best_val = sa_val
            best_pts = sa_pts.copy()
    
    # Final local refinement on the best result
    if best_pts is not None and time.time() < _deadline - 5.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=25, step0=0.015, deadline=_deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
