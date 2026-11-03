"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """Compute the exact covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid if hull fails (should be rare for N=50)
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
        # The circumcenter of a spherical triangle is the point on the sphere equidistant from the vertices.
        # It lies on the normal to the plane containing a,b,c.
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            # Degenerate triangle, use max edge length / 2
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        # Ensure u points away from the origin (outward normal)
        # The normal to the plane a,b,c is n = (b-a) x (c-a).
        # If n . a > 0, the origin is on the opposite side, so u or -u is the outward normal.
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=200):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _soft_covering_radius(pts, sample_pts, beta):
    """
    Compute the soft covering radius using log-sum-exp.
    This provides a smooth approximation to the max distance.
    """
    # pts: (N, 3)
    # sample_pts: (M, 3)
    # dots: (M, N)
    dots = sample_pts @ pts.T
    dots = np.clip(dots, -1, 1)
    
    # Distance squared approx: 2 * (1 - cos(theta))
    # We want to minimize max distance.
    # Let's use the angle directly for better scaling.
    angles = np.arccos(dots)
    
    # Soft max
    max_angles = np.max(angles, axis=1)
    # To avoid numerical issues with large beta, we can use:
    # log(sum(exp(beta * (angle - max_angle)))) + max_angle
    # But standard log-sum-exp is fine if we subtract the max.
    scaled = beta * angles
    max_scaled = np.max(scaled, axis=1)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled[:, np.newaxis]), axis=1))
    soft_radius = log_sum_exp / beta
    
    return np.mean(soft_radius) # Or max? Mean is smoother, max is closer to objective.
    # Actually, minimizing the mean of the soft-max over the grid is a good surrogate.
    # Let's return the mean to make the gradient more stable.

def _objective_and_grad(flat_pts, sample_pts, beta):
    """
    Objective: soft covering radius using log-sum-exp.
    Returns (obj, grad) where grad is with respect to flat_pts (unnormalized).
    
    The key insight: we optimize over flat_pts, but the objective depends on
    normalized points. We need the full chain rule:
      pts = flat_pts / ||flat_pts||
      f(flat_pts) = g(pts(flat_pts))
      grad_flat = J^T * grad_pts
    where J is the Jacobian of pts w.r.t. flat_pts.
    """
    n_pts = len(flat_pts) // 3
    flat_pts = flat_pts.reshape(n_pts, 3)
    norms = np.linalg.norm(flat_pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = flat_pts / norms
    
    dots = sample_pts @ pts.T
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)
    
    scaled = beta * angles
    max_scaled = np.max(scaled, axis=1)
    exp_scaled = np.exp(scaled - max_scaled[:, np.newaxis])
    sum_exp = np.sum(exp_scaled, axis=1)
    
    # Soft radius for each sample point
    soft_r = (max_scaled + np.log(sum_exp)) / beta
    
    # Objective: mean soft radius
    obj = np.mean(soft_r)
    
    # Gradient of soft_r w.r.t. angles:
    # d(soft_r)/d(angle_j) = exp(beta * angle_j) / sum_k exp(beta * angle_k)
    weights = exp_scaled / sum_exp[:, np.newaxis]  # (M, N)
    
    # d(angle)/d(dot) = -1 / sqrt(1 - dot^2)
    denom = np.sqrt(1 - dots**2 + 1e-12)
    grad_angles = -weights / denom  # (M, N)
    
    # d(dot)/d(pts) = sample_pts^T, so:
    # grad_pts[i] = sum_j grad_angles[j, i] * sample_pts[j]
    grad_pts = grad_angles.T @ sample_pts  # (N, 3)
    
    # Now chain rule through normalization:
    # pts = flat / ||flat||
    # d(pts_i)/d(flat_k) = delta_ik / ||flat_i|| - flat_ik * flat_ij / ||flat_i||^3
    #                     = (delta_ik - pts_ik * pts_ij) / ||flat_i||
    # So: grad_flat[i, k] = sum_j grad_pts[i, j] * (delta_jk - pts_ij * pts_ik) / ||flat_i||
    #                       = (grad_pts[i, k] - pts_ik * sum_j grad_pts[i, j] * pts_ij) / ||flat_i||
    #                       = (grad_pts[i, k] - pts_ik * (grad_pts[i] . pts[i])) / ||flat_i||
    
    dot_grad_pts = np.sum(grad_pts * pts, axis=1, keepdims=True)  # (N, 1)
    grad_flat = (grad_pts - dot_grad_pts * pts) / norms  # (N, 3)
    
    return obj, grad_flat.ravel()

def _optimize_smooth(pts, sample_pts, beta=100.0, max_iter=2000):
    """
    Optimize the configuration using L-BFGS-B on the smooth surrogate.
    """
    flat_pts = pts.reshape(-1)
    
    res = minimize(
        _objective_and_grad,
        flat_pts,
        args=(sample_pts, beta),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    return res.x.reshape(-1, 3)

def _polish_exact(pts, max_iter=500, deadline=None):
    """
    Final polish using exact circumradius minimization via local moves.
    Uses adaptive step size and checks deadline.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        max_cr = 0.0
        max_facet = None
        max_circum_center = None
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            n_face = n_face / np.linalg.norm(n_face)
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            if cr > max_cr:
                max_cr = cr
                max_facet = simplex
                max_circum_center = cc
                
        if max_facet is None:
            break
            
        # Adaptive step size: start large, reduce if not improving
        step = 0.15 * min(1.0, max_cr)
        for idx in max_facet:
            p = pts[idx]
            vec = max_circum_center - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                pts[idx] = p + step * (vec_tan / norm_vec)
                
        pts = _normalize(pts)
        
        if max_cr < 1e-9:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate sample points for the soft objective
    # Use a Fibonacci sphere for good coverage - more samples for better gradient
    n_samples = 8000
    sample_pts = _fibonacci_points(n_samples)
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius_exact(best_pts)
    
    np.random.seed(42)
    
    # Multi-start strategy with more diverse starts
    starts = [
        _fibonacci_points(N),
        _normalize(np.random.randn(N, 3)),
        _normalize(_fibonacci_points(N) + 0.15 * np.random.randn(N, 3)),
        _normalize(_fibonacci_points(N) + 0.3 * np.random.randn(N, 3)),
        # Antipodal pairs + random
        _normalize(np.vstack([
            np.random.randn(25, 3),
            -np.random.randn(25, 3)
        ]))
    ]
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 25:
            break
            
        current_pts = start_pts.copy()
        
        # Smooth optimization with annealing: start coarse, get finer
        # Phase 1: Coarse (low beta, more iterations)
        pts_smooth = _optimize_smooth(current_pts, sample_pts, beta=8.0, max_iter=800)
        
        if time.time() > deadline - 20:
            pts_final = _polish_exact(pts_smooth, max_iter=100, deadline=deadline)
            cr = _covering_radius_exact(pts_final)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts_final.copy()
            continue
        
        # Phase 2: Medium (higher beta)
        pts_smooth = _optimize_smooth(pts_smooth, sample_pts, beta=30.0, max_iter=600)
        
        if time.time() > deadline - 15:
            pts_final = _polish_exact(pts_smooth, max_iter=100, deadline=deadline)
            cr = _covering_radius_exact(pts_final)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts_final.copy()
            continue
        
        # Phase 3: Fine (very high beta)
        pts_smooth = _optimize_smooth(pts_smooth, sample_pts, beta=80.0, max_iter=800)
        
        # Phase 4: Exact polish with deadline check
        pts_final = _polish_exact(pts_smooth, max_iter=300, deadline=deadline)
        
        cr = _covering_radius_exact(pts_final)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts_final.copy()
            
        # If time allows, try perturbation of best found
        if time.time() < deadline - 35 and i >= 1:
            noise_scale = 0.08
            noise = noise_scale * np.random.randn(N, 3)
            perturbed = _normalize(best_pts + noise)
            pts_smooth_p = _optimize_smooth(perturbed, sample_pts, beta=40.0, max_iter=500)
            pts_final_p = _polish_exact(pts_smooth_p, max_iter=150, deadline=deadline)
            cr_p = _covering_radius_exact(pts_final_p)
            if cr_p < best_cr:
                best_cr = cr_p
                best_pts = pts_final_p.copy()

    # Final polish on best if time remains
    if time.time() < deadline - 10:
        pts_final = _polish_exact(best_pts, max_iter=200, deadline=deadline)
        cr_final = _covering_radius_exact(pts_final)
        if cr_final < best_cr:
            best_cr = cr_final
            best_pts = pts_final

    return _normalize(best_pts)
# EVOLVE-BLOCK-END
