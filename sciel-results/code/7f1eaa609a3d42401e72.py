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
    pts = flat_pts.reshape(-1, 3)
    pts = _normalize(pts)
    
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
    
    # Gradient
    # d(soft_r)/d(angle) = exp(beta*angle) / sum(exp(beta*angle_j))
    weights = exp_scaled / sum_exp[:, np.newaxis]
    
    # d(angle)/d(dot) = -1 / sqrt(1 - dot^2)
    # d(dot)/d(pts) = -sample_pts^T
    # d(pts)/d(flat_pts) = 1 (before normalization)
    
    # Note: We are optimizing on the sphere, so we need to project the gradient.
    # But L-BFGS-B works on the flat representation.
    # The gradient with respect to the unnormalized points is:
    # grad_flat = sum_j weights[j] * (-sample_pts[j]) / sqrt(1 - dot_j^2)
    
    denom = np.sqrt(1 - dots**2 + 1e-12)
    grad_angles = -weights / denom
    
    # grad_dots: (M, N)
    # grad_pts: (N, 3)
    # grad_pts[i] = sum_j grad_dots[j, i] * sample_pts[j]
    grad_pts = (grad_angles * 1.0) .T @ sample_pts
    
    # Normalize gradient to account for the constraint |pts|=1?
    # Actually, if we use a parameterization that stays on the sphere, we don't need this.
    # But we are using flat coordinates.
    # The projection of the gradient onto the tangent space is:
    # grad_tan = grad_pts - (grad_pts . pts) * pts
    # However, L-BFGS-B will try to move in the direction of grad_pts.
    # If we just use grad_pts, it might move off the sphere.
    # A better approach is to use a retraction or just normalize in the objective.
    # Since we normalize in the objective, the gradient is technically incorrect for the flat space.
    # We need to chain rule:
    # pts = flat / ||flat||
    # d(pts_i)/d(flat_k) = delta_ik / ||flat|| - flat_i flat_k / ||flat||^3
    
    # Let's compute the correct gradient for the normalized points.
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # pts = flat / norms
    # grad_flat = (I - pts pts^T) @ grad_pts
    # This projects the gradient onto the tangent space of the sphere at pts.
    grad_flat = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
    
    return obj, grad_flat

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

def _polish_exact(pts, max_iter=500):
    """
    Final polish using exact circumradius minimization via local moves.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
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
            
        # Move points in the worst facet towards the circumcenter
        for idx in max_facet:
            p = pts[idx]
            vec = max_circum_center - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                step = 0.1 * min(1.0, max_cr)
                pts[idx] = p + step * (vec_tan / norm_vec)
                
        pts = _normalize(pts)
        
        if max_cr < 1e-8:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate sample points for the soft objective
    # Use a Fibonacci sphere for good coverage
    n_samples = 5000
    sample_pts = _fibonacci_points(n_samples)
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius_exact(best_pts)
    
    np.random.seed(42)
    
    # Multi-start strategy
    # 1. Fibonacci
    # 2. Random
    # 3. Perturbed Fibonacci
    
    starts = [
        _fibonacci_points(N),
        _normalize(np.random.randn(N, 3)),
        _normalize(_fibonacci_points(N) + 0.2 * np.random.randn(N, 3))
    ]
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 20:
            break
            
        # Smooth optimization
        # Start with a lower beta and increase it (annealing)
        # Or just use a fixed high beta.
        # Let's try a simple annealing schedule.
        current_pts = start_pts.copy()
        
        # Phase 1: Coarse optimization
        pts_smooth = _optimize_smooth(current_pts, sample_pts, beta=10.0, max_iter=500)
        # Phase 2: Fine optimization
        pts_smooth = _optimize_smooth(pts_smooth, sample_pts, beta=50.0, max_iter=500)
        # Phase 3: Very fine optimization
        pts_smooth = _optimize_smooth(pts_smooth, sample_pts, beta=100.0, max_iter=1000)
        
        # Phase 4: Exact polish
        pts_final = _polish_exact(pts_smooth, max_iter=200)
        
        cr = _covering_radius_exact(pts_final)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts_final.copy()
            
        # If time allows, try a perturbation of the best found so far
        if time.time() < deadline - 30 and i > 0:
             noise = 0.1 * np.random.randn(N, 3)
             perturbed = _normalize(best_pts + noise)
             pts_smooth_p = _optimize_smooth(perturbed, sample_pts, beta=50.0, max_iter=500)
             pts_final_p = _polish_exact(pts_smooth_p, max_iter=200)
             cr_p = _covering_radius_exact(pts_final_p)
             if cr_p < best_cr:
                 best_cr = cr_p
                 best_pts = pts_final_p.copy()

    return _normalize(best_pts)
# EVOLVE-BLOCK-END
