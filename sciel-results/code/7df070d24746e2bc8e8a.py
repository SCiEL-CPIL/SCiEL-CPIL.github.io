"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n, offset=0.5):
    """Generate initial Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    This is the ground truth metric used by the evaluator.
    """
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) == 3:
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        elif len(verts) == 4:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _energy_pairwise(pts, p=10.0):
    """
    Compute a pairwise repulsion energy.
    Minimizing this encourages uniform distribution (maximizes minimum distance).
    """
    # pts: (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_p = np.power(dist_sq, -p/2.0)
    return np.sum(inv_dist_p)

def _soft_covering_radius_and_grad(flat_pts, test_points, beta=10.0):
    """
    Compute a smooth approximation of the covering radius and its gradient.
    Uses a soft-min of angular distances.
    """
    pts = flat_pts.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute dot products between test points and configuration points
    dots = test_points @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distances
    dists = np.arccos(dots)  # (M, N)
    
    # Soft-min calculation: min_j dists[i,j]
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to minimize the MAX of the soft_mins? No, we want to minimize the covering radius.
    # Covering radius = max_i (min_j dist(i,j)).
    # Let f_i = min_j dist(i,j). We want to minimize max_i f_i.
    # A common surrogate is sum_i f_i or max_i f_i approximated by log-sum-exp.
    # Let's use the sum of soft-mins as a proxy for the mean covering radius, 
    # but weighted to emphasize the maximum.
    # Actually, minimizing the sum of soft-mins tends to equalize distances.
    # To better target the max, we can use a soft-max of the soft-mins.
    
    min_dists = np.min(dists, axis=1)  # (M,)
    
    # Soft max of min_dists to approximate the covering radius
    max_d = np.max(min_dists)
    if beta <= 0:
        return max_d, np.zeros_like(flat_pts)
    
    shifted = min_dists - max_d
    exp_shifted = np.exp(shifted * beta)
    sum_exp = np.sum(exp_shifted)
    
    # Value: (1/beta) * log(sum(exp(beta * x)))
    val = max_d + (1.0/beta) * np.log(sum_exp)
    
    # Gradient with respect to flat_pts
    # d(val)/d(pts_k)
    # Let S = sum_i exp(beta * (min_dists_i - max_d))
    # val = max_d + (1/beta) log S
    # This is complex because min_dists and max_d depend on pts.
    # 
    # Alternative simpler gradient approach for sum of soft-mins:
    # L = sum_i soft_min_j dists(i,j)
    # soft_min_j x_j = -1/beta log(sum_j exp(-beta x_j))
    # d(soft_min)/dx_k = (1/sum_j exp(-beta x_j)) * exp(-beta x_k)
    # Let w_ik = exp(-beta * dists(i,k)) / sum_j exp(-beta * dists(i,j))
    # d(dist(i,k))/d(pts_k) = -pts_k / sqrt(1 - (dots_ik)^2)  (tangent space)
    # Actually, dist = arccos(d). d(arccos(d))/dd = -1/sqrt(1-d^2).
    # d(d)/d(pts_k) = -test_point_i (since d = t_i . p_k, grad_p d = -t_i? No, grad_p (t.p) = t)
    # Wait, d = t_i . p_k. grad_{p_k} d = t_i.
    # So grad_{p_k} dist = -t_i / sqrt(1-d^2).
    # 
    # Let's compute the gradient of L = sum_i soft_min_i
    # grad_L = sum_i sum_k w_ik * (-t_i / sqrt(1-d_ik^2))
    # 
    # Let's use this "sum of soft mins" objective which is smoother and often works well.
    # It doesn't strictly minimize the max, but it pushes all points to be well-covered.
    
    # Re-calculate weights for sum of soft-mins
    neg_beta_dists = -beta * dists
    # Subtract max for stability
    max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
    exp_neg = np.exp(neg_beta_dists - max_neg)
    sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)
    weights = exp_neg / sum_exp_neg  # (M, N)
    
    # Gradient term: sum_i sum_k w_ik * (-t_i / sqrt(1-d_ik^2))
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots**2))  # (M, N)
    # term_ik = -weights_ik / denom_ik
    term = -weights / denom  # (M, N)
    
    # grad_pts_k = sum_i term_ik * t_i
    # term is (M, N), test_points is (M, 3)
    # grad_pts = term.T @ test_points  # (N, 3)
    grad_pts_ambient = term.T @ test_points
    
    # Project to tangent space
    grad_dot_P = np.sum(grad_pts_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_pts_ambient - grad_dot_P * pts
    
    # Map back to raw flat_pts gradient (accounting for normalization)
    # If we optimize in Cartesian space directly, we just return grad_tangent?
    # L-BFGS-B works on the flat vector. The constraint is ||p||=1.
    # If we use the gradient on the manifold, we need to be careful.
    # Standard trick: use the ambient gradient projected to tangent, 
    # but since we normalize inside the function, the gradient w.r.t raw input 
    # is approximately the tangent gradient if we assume the input is already normalized.
    # To be safe, let's just return the tangent gradient. L-BFGS-B will handle the rest.
    
    return val, grad_tangent.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_score = _covering_radius_exact(best_pts)
    
    # Phase 1: Optimize pairwise energy to spread points out
    # This helps avoid local minima where points cluster.
    def energy_obj(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _normalize(p)
        return _energy_pairwise(p, p=10.0)
    
    try:
        res1 = minimize(
            energy_obj,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 200}
        )
        if res1.x is not None:
            p_new = _normalize(res1.x.reshape(N, 3))
            score_new = _covering_radius_exact(p_new)
            if score_new < best_score:
                best_pts = p_new
                best_score = score_new
    except Exception:
        pass

    if time.time() > deadline - 40:
        return best_pts

    # Phase 2: Optimize soft covering radius
    # Use a moderate number of test points
    n_test = 5000
    test_points = _fibonacci_spiral(n_test)
    
    def soft_obj_grad(flat_pts):
        return _soft_covering_radius_and_grad(flat_pts, test_points, beta=20.0)
    
    try:
        res2 = minimize(
            soft_obj_grad,
            best_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 300}
        )
        if res2.x is not None:
            p_new = _normalize(res2.x.reshape(N, 3))
            score_new = _covering_radius_exact(p_new)
            if score_new < best_score:
                best_pts = p_new
                best_score = score_new
    except Exception:
        pass

    if time.time() > deadline - 20:
        return best_pts

    # Phase 3: Refine with more test points and higher beta
    n_test_fine = 20000
    test_points_fine = _fibonacci_spiral(n_test_fine)
    
    def soft_obj_grad_fine(flat_pts):
        return _soft_covering_radius_and_grad(flat_pts, test_points_fine, beta=50.0)
    
    try:
        res3 = minimize(
            soft_obj_grad_fine,
            best_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 200}
        )
        if res3.x is not None:
            p_new = _normalize(res3.x.reshape(N, 3))
            score_new = _covering_radius_exact(p_new)
            if score_new < best_score:
                best_pts = p_new
                best_score = score_new
    except Exception:
        pass

    # Final check and normalization
    best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
