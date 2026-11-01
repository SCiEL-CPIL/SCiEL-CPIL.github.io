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

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sample(n, seed=None):
    """Generate n random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii.
    Returns facets (indices), circumcenters (unit vectors), radii (radians).
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    # Filter out degenerate facets
    valid = norm_n[:, 0] > 1e-15
    if not np.all(valid):
        a = a[valid]
        b = b[valid]
        c = c[valid]
        n = n[valid]
        norm_n = norm_n[valid]
        facets = facets[valid]
    
    if len(facets) == 0:
        return None, None, None
    
    n = n / norm_n
    
    # Ensure outward orientation
    mid = (a + b + c) / 3.0
    dots = np.sum(n * mid, axis=1)
    n = np.where(dots[:, np.newaxis] < 0, -n, n)
    
    # Circumradius: angular distance from circumcenter to any vertex
    cos_r = np.clip(np.sum(n * a, axis=1), -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    return facets, n, radii

def _covering_radius_deg_hull(p):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(p)
    if facets is None:
        return 180.0
    if len(radii) == 0:
        return 180.0
    return np.degrees(np.max(radii))

def _soft_max_objective(params, deadline, temperature=0.05):
    """
    Soft-max objective: approximates the maximum facet radius using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    if time.time() > deadline:
        return 0.0
    
    p = params.reshape(N, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    p = p / norms
    
    facets, circumcenters, radii = _compute_hull_data(p)
    if facets is None or len(radii) == 0:
        return 1.0  # Large penalty
    
    # Log-sum-exp for soft max
    max_r = np.max(radii)
    if max_r < 1e-15:
        return 0.0
    
    # Soft max: max_r + temperature * log(sum(exp((r - max_r)/temperature)))
    shifted = (radii - max_r) / temperature
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = max_r + temperature * log_sum
    
    return soft_max

def _soft_max_grad(params, deadline, temperature=0.05, eps=1e-5):
    """
    Numerical gradient of the soft-max objective.
    """
    if time.time() > deadline:
        return np.zeros_like(params)
    
    grad = np.zeros_like(params)
    f0 = _soft_max_objective(params, deadline, temperature)
    
    for i in range(len(params)):
        orig = params[i]
        params[i] += eps
        f1 = _soft_max_objective(params, deadline, temperature)
        params[i] = orig
        
        grad[i] = (f1 - f0) / eps
    
    return grad

def _repulsion_relax(config, deadline, n_iter=100, strength=0.1):
    """
    Simple repulsion-based relaxation to spread points.
    """
    points = config.copy()
    
    for _ in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        # Compute repulsion forces
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.dot(points[i], points[j])
                if d < 0.99:  # Only repel when close
                    forces[i] -= strength * points[j]
        
        # Project to tangent plane
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + 0.01 * forces_tan
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points

def _lloyd_iteration(points, deadline):
    """
    Single Lloyd iteration: move each point toward the centroid of its Voronoi cell.
    Approximated by moving toward the circumcenter of facets it belongs to.
    """
    if time.time() > deadline - 1:
        return points
    
    facets, circumcenters, radii = _compute_hull_data(points)
    if facets is None:
        return points
    
    forces = np.zeros_like(points)
    
    for k in range(len(facets)):
        tri = facets[k]
        cc = circumcenters[k]
        r = radii[k]
        
        # Weight by radius (larger facets pull harder)
        weight = r
        
        for idx in tri:
            forces[idx] += weight * (cc - points[idx])
    
    # Project to tangent plane
    dot_fp = np.sum(forces * points, axis=1)
    forces_tan = forces - dot_fp[:, np.newaxis] * points
    
    # Adaptive step size
    step = 0.1
    points = points + step * forces_tan
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points

def _refine_lloyd(config, deadline):
    """
    Multi-stage Lloyd refinement with decreasing step size.
    """
    points = config.copy()
    
    best_points = points.copy()
    best_score = _covering_radius_deg_hull(points)
    
    # Stages: (step_scale, max_iter)
    stages = [
        (0.5, 30),
        (0.2, 50),
        (0.1, 80),
        (0.05, 120),
        (0.02, 200),
        (0.01, 300),
    ]
    
    for scale, iters in stages:
        if time.time() > deadline - 5:
            break
        
        for _ in range(iters):
            if time.time() > deadline - 2:
                break
            
            new_points = _lloyd_iteration(points, deadline)
            
            # Check if improved
            score = _covering_radius_deg_hull(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points.copy()
                points = new_points.copy()
            else:
                # Revert
                points = best_points.copy()
                break
    
    return best_points, best_score

def _projected_gradient_polish(config, deadline, temperature=0.02, n_iter=50):
    """
    Final polish using projected gradient descent on the soft-max objective.
    """
    params = config.flatten().copy()
    best_params = params.copy()
    best_score = _covering_radius_deg_hull(config)
    
    lr = 0.01
    
    for i in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        f = _soft_max_objective(params, deadline, temperature)
        grad = _soft_max_grad(params, deadline, temperature)
        
        # Gradient step
        params = params - lr * grad
        
        # Normalize each point
        p = params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        params = p.flatten()
        
        score = _covering_radius_deg_hull(p)
        if score < best_score:
            best_score = score
            best_params = params.copy()
        
        # Decay learning rate
        lr *= 0.95
    
    best_config = best_params.reshape(N, 3)
    norms = np.linalg.norm(best_config, axis=1, keepdims=True)
    best_config = best_config / norms
    
    return best_config, best_score

def _lbfgs_stage(config, deadline, temperature=0.03, max_iter=200):
    """
    L-BFGS-B optimization on the soft-max objective.
    """
    x0 = config.flatten()
    bounds = [(-2.0, 2.0)] * (N * 3)
    
    def obj(x):
        if time.time() > deadline:
            return 0.0
        return _soft_max_objective(x, deadline, temperature)
    
    def grad(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        return _soft_max_grad(x, deadline, temperature)
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
        
        if res.success:
            p_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            p_opt = p_opt / norms
            return p_opt
    except Exception:
        pass
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_config = None
    best_score = float('inf')
    
    # Initial configurations
    inits = []
    inits.append(_fib_sample(N))
    for i in range(8):
        inits.append(_random_sample(N, seed=100 + i))
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # 1. Repulsion relaxation
        relaxed = _repulsion_relax(init, deadline - 25, n_iter=50, strength=0.05)
        
        # 2. Lloyd refinement
        refined_l, score_l = _refine_lloyd(relaxed, deadline - 20)
        
        if score_l < best_score:
            best_score = score_l
            best_config = refined_l.copy()
        
        # 3. L-BFGS on soft-max
        if time.time() < deadline - 15:
            refined_lb = _lbfgs_stage(refined_l, deadline - 10, temperature=0.03, max_iter=100)
            score_lb = _covering_radius_deg_hull(refined_lb)
            
            if score_lb < best_score:
                best_score = score_lb
                best_config = refined_lb.copy()
        
        # 4. Gradient polish
        if time.time() < deadline - 5:
            refined_g, score_g = _projected_gradient_polish(best_config, deadline - 2, temperature=0.01, n_iter=30)
            
            if score_g < best_score:
                best_score = score_g
                best_config = refined_g.copy()
        
        # Early stop if very good
        if best_score < 18.31:
            break
    
    # Final polish with tighter tolerance
    if best_config is not None and time.time() < deadline - 3:
        final_config, final_score = _projected_gradient_polish(best_config, deadline - 1, temperature=0.005, n_iter=50)
        if final_score < best_score:
            best_score = final_score
            best_config = final_config
    
    if best_config is None:
        best_config = _fib_sample(N)
    
    # Final normalization
    best_config = np.array([_normalize(p) for p in best_config])
    
    return best_config
