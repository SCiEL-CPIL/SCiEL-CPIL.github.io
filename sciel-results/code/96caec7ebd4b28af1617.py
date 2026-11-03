"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_facet_circumradii(pts):
    """
    Compute angular circumradii for all facets of the convex hull.
    Returns an array of circumradii (in radians) and the hull simplices.
    Returns (np.array([np.pi]), None) if invalid.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Vectorized coincident point check
    dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        return np.array([np.pi]), None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi]), None
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.array([np.pi]), None
        
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Normal to each facet
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = (norms[:, 0] > 1e-12)
    if not np.all(valid):
        return np.array([np.pi]), None
    
    normals = normals[valid] / norms[valid]
    a_v = a[valid]
    b_v = b[valid]
    c_v = c[valid]
    
    # For each facet, compute angular distance from normal to each vertex
    # The circumcenter is either +n or -n; pick the one with smaller max distance
    dot_a = np.clip(normals @ a_v.T, -1, 1)
    dot_b = np.clip(normals @ b_v.T, -1, 1)
    dot_c = np.clip(normals @ c_v.T, -1, 1)
    
    # Angular distances for +n
    da_pos = np.arccos(dot_a)
    db_pos = np.arccos(dot_b)
    dc_pos = np.arccos(dot_c)
    max_pos = np.maximum(np.maximum(da_pos, db_pos), dc_pos)
    
    # Angular distances for -n
    da_neg = np.arccos(-dot_a)
    db_neg = np.arccos(-dot_b)
    dc_neg = np.arccos(-dot_c)
    max_neg = np.maximum(np.maximum(da_neg, db_neg), dc_neg)
    
    # For each facet, the circumradius is min(max_pos, max_neg)
    # We also need to know which side (+ or -) was chosen for gradient calculation
    # is_pos = max_pos <= max_neg
    facet_radii = np.minimum(max_pos, max_neg)
    
    return facet_radii, simplices[valid]

def _covering_radius_fast(pts):
    """
    Fast covering radius computation via convex hull facet circumradii.
    Returns the covering radius in radians, or np.pi if invalid.
    """
    radii, _ = _compute_facet_circumradii(pts)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def _smooth_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimization.
    Returns value and gradient.
    """
    pts = _to_sphere(pts)
    radii, simplices = _compute_facet_circumradii(pts)
    
    if len(radii) == 0 or simplices is None:
        return np.pi, np.zeros_like(pts)
    
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(pts)
    
    shifted = radii - max_r
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_val = max_r + log_sum
    
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(pts)
    
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    dot_a = np.clip(normals @ a.T, -1, 1)
    dot_b = np.clip(normals @ b.T, -1, 1)
    dot_c = np.clip(normals @ c.T, -1, 1)
    
    da_pos = np.arccos(dot_a)
    db_pos = np.arccos(dot_b)
    dc_pos = np.arccos(dot_c)
    
    da_neg = np.arccos(-dot_a)
    db_neg = np.arccos(-dot_b)
    dc_neg = np.arccos(-dot_c)
    
    max_pos = np.maximum(np.maximum(da_pos, db_pos), dc_pos)
    max_neg = np.maximum(np.maximum(da_neg, db_neg), dc_neg)
    
    is_pos = max_pos <= max_neg
    
    poles = np.where(is_pos[:, np.newaxis], normals, -normals)
    
    pos_max_idx = np.argmax(np.column_stack([da_pos, db_pos, dc_pos]), axis=1)
    neg_max_idx = np.argmax(np.column_stack([da_neg, db_neg, dc_neg]), axis=1)
    
    active_indices = np.where(is_pos, pos_max_idx, neg_max_idx)
    
    active_point_indices = simplices[np.arange(len(simplices)), active_indices]
    active_points = pts[active_point_indices]
    
    active_dots = np.einsum('ij,ij->i', poles, active_points)
    active_dots = np.clip(active_dots, -1, 1)
    
    denom = np.sqrt(1.0 - active_dots * active_dots)
    denom[denom < 1e-8] = 1e-8
    
    grad_vecs = -poles / denom[:, np.newaxis]
    
    weighted_grads = weights[:, np.newaxis] * grad_vecs
    
    for i in range(len(simplices)):
        idx = active_point_indices[i]
        grad[idx] += weighted_grads[i]
        
    return smooth_val, grad

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_optimize(pts, max_iter=200, step_size=0.02, deadline=None):
    """
    Local optimization with adaptive step size and multi-point perturbations.
    """
    best_pts = pts.copy()
    best_val = _covering_radius_fast(best_pts)
    current_step = step_size
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb 1-3 random points simultaneously
        n_perturb = np.random.choice([1, 1, 2, 3])
        indices = np.random.choice(len(best_pts), size=n_perturb, replace=False)
        
        new_pts = best_pts.copy()
        for idx in indices:
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            new_pts[idx] = best_pts[idx] + current_step * direction
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius_fast(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts
            best_val = new_val
            current_step = min(current_step * 1.02, 0.1)
        else:
            current_step *= 0.93
            
        if current_step < 1e-6:
            break
    
    return best_pts, best_val

def _gradient_refine(pts, max_iter=80, deadline=None):
    """
    Gradient-based refinement using the smooth surrogate with multiple beta values.
    """
    current_pts = pts.copy()
    current_val = _covering_radius_fast(current_pts)
    
    # Try different beta values for the smooth surrogate
    for beta in [40.0, 80.0, 150.0]:
        if deadline is not None and time.time() > deadline:
            break
        
        def obj_and_grad(flat_pts):
            pts_local = _to_sphere(flat_pts.reshape(N, 3))
            val, grad = _smooth_covering_radius(pts_local, beta=beta)
            return val, grad.flatten()
            
        try:
            result = minimize(obj_and_grad, current_pts.flatten(), method='L-BFGS-B', jac=True,
                              options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12})
            new_pts = _to_sphere(result.x.reshape(N, 3))
            new_val = _covering_radius_fast(new_pts)
            
            if new_val < current_val:
                current_pts = new_pts
                current_val = new_val
        except Exception:
            pass
    
    return current_pts, current_val

def _random_points(n, seed=None):
    """Generate n random points on the unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    return _to_sphere(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses multi-start with Fibonacci spiral and random initializations.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_pts = None
    best_val = np.pi
    
    # Multi-start: try several initial configurations
    initializations = []
    
    # Strategy 1: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    initializations.append(pts)
    
    # Strategy 2: Random points with different seeds
    for seed in [42, 123, 456]:
        if time.time() > deadline - 30.0:
            break
        pts = _random_points(N, seed=seed)
        initializations.append(pts)
    
    # Optimize each initialization
    for init_pts in initializations:
        if time.time() > deadline - 10.0:
            break
        
        # Local optimization
        opt_pts, opt_val = _local_optimize(init_pts, max_iter=150, step_size=0.03, deadline=deadline)
        if opt_val < best_val:
            best_val = opt_val
            best_pts = opt_pts.copy()
        
        # Gradient refinement if time permits
        if time.time() < deadline - 15.0:
            grad_pts, grad_val = _gradient_refine(best_pts, max_iter=60, deadline=deadline)
            if grad_val < best_val:
                best_val = grad_val
                best_pts = grad_pts.copy()
    
    # Final local polish
    if time.time() < deadline - 3.0 and best_pts is not None:
        final_pts, final_val = _local_optimize(best_pts, max_iter=100, step_size=0.005, deadline=deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
