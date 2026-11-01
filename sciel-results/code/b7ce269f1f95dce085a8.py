import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _normalize_batch(pts):
    """Normalize batch of vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii (vectorized).
    Returns facets, circumcenters, radii.
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    # Vectorized circumcenter computation
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1)  # (F,)
    
    # Handle degenerate facets
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        # Fallback to sampling if hull fails
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    return np.degrees(np.max(radii))

def _soft_max_objective(params, beta=100.0):
    """
    Soft-max surrogate: approximates max(radii) via log-sum-exp.
    As beta -> inf, this approaches the true max.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    r_max = np.max(radii)
    if r_max < 1e-15:
        return r_max
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    return log_sum_exp

def _soft_max_grad(params, beta=100.0):
    """
    Numerical gradient of soft-max objective.
    """
    eps = 1e-6
    grad = np.zeros_like(params)
    base_val = _soft_max_objective(params, beta)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _soft_max_objective(params_plus, beta)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        val_minus = _soft_max_objective(params_minus, beta)
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return grad

def _repulsion_relax(points, alpha_init=0.05, decay=0.995, max_iter=300, deadline=None):
    """
    Repulsion Relaxation (Thomson Problem solver).
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane: F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + alpha * forces_tan
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay alpha
        alpha *= decay
        
    return points

def _lloyd_step_vectorized(config):
    """
    Perform one Lloyd iteration (vectorized).
    Moves each point towards the spherical mean of the circumcenters of the faces for which it is the nearest site.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return config
    
    facets = hull.simplices
    a = config[facets[:, 0]]
    b = config[facets[:, 1]]
    c = config[facets[:, 2]]
    
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    valid = norm_n > 1e-12
    if not np.any(valid):
        return config
    
    cc = n[valid] / norm_n[valid, np.newaxis]
    dots = np.sum(cc * a[valid], axis=1)
    cc[dots < 0] *= -1
    
    # For each circumcenter, find the nearest site
    dots_all = cc @ config.T
    nearest_idx = np.argmax(dots_all, axis=1)
    
    # Accumulate targets for each site
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    
    np.add.at(sums, nearest_idx, cc)
    np.add.at(counts, nearest_idx, 1)
    
    new_config = config.copy()
    for i in range(N):
        if counts[i] > 0:
            target = sums[i]
            n = np.linalg.norm(target)
            if n > 1e-10:
                new_config[i] = target / n
            else:
                new_config[i] = config[i]
        else:
            new_config[i] = config[i]
            
    return new_config

def _projected_gradient_polish(config, deadline, lr_init=0.01, lr_decay=0.98, max_iter=200):
    """
    Final polish using projected gradient descent on the tangent space.
    Directly minimizes the max angular distance.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Compute gradient of max radius w.r.t. each point
        facets, circumcenters, radii = _compute_hull_data(config)
        if facets is None:
            break
        
        # Find which facet has the max radius
        max_idx = np.argmax(radii)
        max_facet = facets[max_idx]
        
        # The circumcenter of the max facet
        cc = circumcenters[max_idx]
        
        # For each vertex of the max facet, compute the gradient contribution
        # Moving a vertex p_i changes the circumcenter
        # Gradient of radius w.r.t. p_i: we want to move p_i to reduce the cap
        # The direction to move is away from the circumcenter
        grad = np.zeros((N, 3))
        
        for vi in max_facet:
            # The vertex vi should move in the direction that shrinks the facet
            # Move away from circumcenter, projected to tangent plane
            dir_vec = config[vi] - cc
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-15)
            # Project to tangent plane at config[vi]
            dot = np.dot(dir_vec, config[vi])
            grad[vi] = dir_vec - dot * config[vi]
        
        # Normalize gradient
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-15:
            break
        grad = grad / g_norm
        
        # Update
        new_config = config - lr * grad
        new_config = _normalize_batch(new_config)
        
        new_score = _covering_radius_deg_hull(new_config)
        
        if new_score < best_score:
            best_score = new_score
            best_config = new_config
            config = new_config
        else:
            lr *= lr_decay
        
    return best_config

def _optimize(config, deadline):
    """
    Full optimization pipeline: Lloyd's + L-BFGS + local optimization + polish.
    """
    config = config.copy()
    
    # Phase 1: Lloyd's algorithm for good initial configuration
    M = 10000
    grid = _fib_sample(M)
    
    for iteration in range(150):
        if time.time() > deadline - 50:
            break
        dots = grid @ config.T
        max_idx = np.argmax(dots, axis=1)
        
        new_config = np.zeros_like(config)
        for i in range(N):
            mask = (max_idx == i)
            count = np.sum(mask)
            if count > 0:
                center = np.mean(grid[mask], axis=0)
                new_config[i] = _normalize(center)
            else:
                new_config[i] = _normalize(np.random.randn(3))
        
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
        
    # Phase 2: L-BFGS Optimization
    if time.time() < deadline - 30:
        config = _optimize_lgbfgs(config, deadline)
        
    # Phase 3: Local optimization
    if time.time() < deadline - 15:
        config = _local_optimize(config, deadline, min_time_remaining=10.0)
        
    # Phase 4: Polish
    if time.time() < deadline - 5:
        config = _polish(config, deadline, min_time_remaining=3.0)
        
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Start with Fibonacci spiral
    config = _fib_sample(N)

    # Run optimization
    config = _optimize(config, deadline)

    # Final normalization and validation
    config = _normalize_batch(config)

    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config

def _optimize_lgbfgs(config, deadline):
    """
    Optimize using L-BFGS-B with a smooth surrogate.
    """
    config = config.copy()
    x0 = config.flatten()
    
    def fun_grad(x):
        cfg = x.reshape(N, 3)
        cfg = _normalize_batch(cfg)
        val = _soft_max_objective(cfg)
        grad = _soft_max_grad(cfg)
        return val, grad.flatten()
    
    best_x = x0
    best_val, _ = fun_grad(x0)
    
    # Run L-BFGS
    while time.time() < deadline - 10:
        res = minimize(fun_grad, best_x, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        else:
            break
            
    best_cfg = _normalize_batch(best_x.reshape(N, 3))
    return best_cfg

def _local_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using circumcenter-based critical points.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.04
    stagnation = 0
    max_stagnation = 30
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        # Find the worst circumcenter
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        worst_idx = np.argmax(min_angles)
        worst_point = circumcenters[worst_idx]
        worst_angle = min_angles[worst_idx]
        
        # Find candidate points to move: the 3 nearest config points to worst_point
        dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots_to_worst)
        
        improved = False
        
        for trial in range(3):
            if trial >= len(sorted_idx):
                break
            idx = sorted_idx[trial]
            
            # Compute tangent direction toward worst_point
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            v_dir = v_proj / v_norm
            
            # Try a few step sizes
            for ss in [step_size, step_size * 0.5, step_size * 2.0]:
                new_pt = config[idx] + ss * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    if new_score < current_score - 0.01:
                        step_size = min(step_size * 1.2, 0.1)
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.85
            if step_size < 0.001 or stagnation > max_stagnation:
                break
    
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish pass with very small steps.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.005
    stagnation = 0
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        # Find all circumcenters with angles close to the maximum
        threshold = np.max(min_angles) - 0.001
        worst_mask = min_angles > threshold
        worst_indices = np.where(worst_mask)[0]
        
        if len(worst_indices) == 0:
            break
        
        improved = False
        
        for wi in worst_indices[:5]:
            worst_point = circumcenters[wi]
            
            dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
            sorted_idx = np.argsort(-dots_to_worst)
            
            for trial in range(2):
                if trial >= len(sorted_idx):
                    break
                idx = sorted_idx[trial]
                
                v = worst_point - config[idx]
                v_proj = v - np.dot(v, config[idx]) * config[idx]
                v_norm = np.linalg.norm(v_proj)
                
                if v_norm < 1e-10:
                    continue
                
                v_dir = v_proj / v_norm
                
                new_pt = config[idx] + step_size * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.0005 or stagnation > 50:
                break
    
    return best_config

def _covering_radius_hull(config):
    """
    Compute covering radius in degrees using convex hull facets.
    Returns (radius_deg, circumcenters_array, facet_indices_array)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    # Vectorized circumcenter computation
    p0 = config[simplices[:, 0]]  # (F, 3)
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0  # (F, 3)
    v2 = p2 - p0  # (F, 3)
    
    # Cross product for each facet
    c = np.cross(v1, v2)  # (F, 3)
    c_norms = np.linalg.norm(c, axis=1)
    
    # Filter out degenerate facets
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return 0.0, np.array([]), np.array([], dtype=int)
    
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    valid_facets = simplices[valid_mask]
    
    # Check if circumcenter is equidistant from all 3 vertices
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])  # (F, 3)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (F, 3)
    
    # Valid circumcenters: all three angles should be approximately equal
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        grid = _fib_sample(10000)
        dots_grid = grid @ config.T
        min_dots = np.min(dots_grid, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    c = c[valid_circ]
    valid_facets = valid_facets[valid_circ]
    
    # For each valid circumcenter, find nearest config point
    dots_all = c @ config.T  # (F, N)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    max_dots = np.max(dots_all, axis=1)  # (F,)
    min_angles = np.arccos(max_dots)  # (F,)
    
    max_angle = np.max(min_angles)
    radius_deg = np.degrees(max_angle)
    
    return radius_deg, c, valid_facets
