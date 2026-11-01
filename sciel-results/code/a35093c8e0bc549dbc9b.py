import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

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

def _spherical_triangle_worst_point(v1, v2, v3):
    """
    Returns the point on the sphere within the spherical triangle (v1, v2, v3)
    that is farthest from the origin? No, farthest from the *centers*?
    We want the point y in the Voronoi cell corresponding to this facet that is
    farthest from the nearest site.
    
    The Voronoi cell vertex is the circumcenter c.
    If c is inside the triangle, the max distance is angle(c, v_i).
    If c is outside, the max distance is at the midpoint of the longest edge.
    
    Returns: (max_angle, worst_point)
    """
    # Compute circumcenter
    # c is proportional to cross(v1-v2, v2-v3)
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    
    if n_norm < 1e-12:
        # Degenerate
        return 0.0, v1

    c = n / n_norm
    
    # Check if c is on the same side of edges as the opposite vertex
    # This determines if the triangle is acute (c inside) or obtuse
    # For spherical triangles, we check dot products
    # If dot(c, v1) > 0, dot(c, v2) > 0, dot(c, v3) > 0, it's likely acute?
    # Actually, for a spherical triangle on the unit sphere, the circumcenter is inside
    # if the triangle is acute.
    
    # Let's check the angles.
    # cos(alpha) = dot(v1, v2)
    # If all angles < pi/2, it's acute.
    
    dot12 = np.dot(v1, v2)
    dot23 = np.dot(v2, v3)
    dot31 = np.dot(v3, v1)
    
    # Clamp for safety
    dot12 = np.clip(dot12, -1.0, 1.0)
    dot23 = np.clip(dot23, -1.0, 1.0)
    dot31 = np.clip(dot31, -1.0, 1.0)
    
    ang12 = np.arccos(dot12)
    ang23 = np.arccos(dot23)
    ang31 = np.arccos(dot31)
    
    # If any angle is >= pi/2, the triangle is obtuse or right.
    # The furthest point is the midpoint of the longest edge.
    # The longest edge corresponds to the largest angle.
    
    max_ang = max(ang12, ang23, ang31)
    
    if max_ang >= np.pi / 2.0:
        # Find the edge corresponding to the max angle
        if max_ang == ang12:
            mid = (v1 + v2) / 2.0
        elif max_ang == ang23:
            mid = (v2 + v3) / 2.0
        else:
            mid = (v3 + v1) / 2.0
        
        mid = _normalize(mid)
        # The distance is half the angle
        dist = max_ang / 2.0
        return dist, mid
    else:
        # Acute triangle, circumcenter is inside
        # Ensure c is in the correct direction (same hemisphere as v1, v2, v3)
        if np.dot(c, v1) < 0:
            c = -c
            
        # Distance is angle(c, v1)
        dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return dist, c

def _compute_covering_radius_and_grads(config):
    """
    Computes the covering radius and gradients for each point.
    Returns: radius (radians), gradients (N, 3)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return np.pi, np.zeros((N, 3))
    
    facets = hull.simplices
    max_radius = 0.0
    grads = np.zeros((N, 3))
    
    for f in facets:
        v1, v2, v3 = config[f[0]], config[f[1]], config[f[2]]
        
        dist, worst_pt = _spherical_triangle_worst_point(v1, v2, v3)
        
        if dist > max_radius:
            max_radius = dist
            
            # Compute gradient contribution
            # We want to move v1, v2, v3 to reduce the distance to worst_pt.
            # The gradient of the distance to a fixed point p from x is (x-p)/||x-p||?
            # On the sphere, we project the direction to the tangent plane.
            
            # Direction from vi to worst_pt
            for i, vi in enumerate([v1, v2, v3]):
                # Vector from vi to worst_pt
                vec = worst_pt - vi
                # Project to tangent plane at vi
                proj = vec - np.dot(vec, vi) * vi
                # Gradient of distance is proportional to -proj (we want to decrease distance)
                # Actually, if we move vi towards worst_pt, the distance decreases?
                # No, if vi is a site, and worst_pt is in its Voronoi cell,
                # moving vi towards worst_pt reduces the distance from worst_pt to vi.
                # So we want to move vi in the direction of worst_pt.
                # Gradient of f = dist(worst_pt, vi) w.r.t vi is - (worst_pt - vi) projected?
                # Let's think: f(vi) = arccos(dot(worst_pt, vi)).
                # d/dvi arccos(dot(w, vi)) = - w / sqrt(1 - dot^2)
                # So gradient is - w * factor.
                # We want to minimize, so we move in direction -grad = w * factor.
                # So we add w (worst_pt) to the gradient?
                # Wait, if we move vi towards w, dot(w, vi) increases, arccos decreases.
                # So the direction of improvement is towards w.
                # So we add the projected vector from vi to w to the "force" that moves vi.
                
                norm_proj = np.linalg.norm(proj)
                if norm_proj > 1e-10:
                    # Normalize to unit step direction
                    grads[f[i]] += proj / norm_proj
    
    # Normalize gradients?
    # The objective is max(dist). The gradient of the max is the gradient of the active term.
    # If multiple facets have the same max radius, we should sum their gradients.
    # For simplicity, we just use the gradient of the single worst facet.
    # A better approach: soft-max.
    
    return max_radius, grads

def _soft_max_objective(params, beta=50.0):
    """
    Smooth surrogate for the covering radius.
    """
    X = params.reshape(N, 3)
    X = _normalize_batch(X)
    
    try:
        hull = ConvexHull(X)
    except Exception:
        return 1e10, np.zeros_like(params)
    
    facets = hull.simplices
    radii = np.zeros(len(facets))
    
    for i, f in enumerate(facets):
        v1, v2, v3 = X[f[0]], X[f[1]], X[f[2]]
        dist, _ = _spherical_triangle_worst_point(v1, v2, v3)
        radii[i] = dist
    
    if len(radii) == 0:
        return 0.0, np.zeros_like(params)
    
    r_max = np.max(radii)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(params)
    
    # Log-sum-exp
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    
    # Gradient
    # d/dx logsumexp(beta * (r - r_max))
    # = (beta * sum(exp(beta*(r_i - r_max)) * grad(r_i)) / sum(exp(beta*(r_i - r_max))))
    # = beta * sum(w_i * grad(r_i)) where w_i = exp(beta*(r_i-r_max)) / sum(...)
    
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(X)
    
    for i, f in enumerate(facets):
        w = weights[i]
        if w < 1e-10:
            continue
            
        v1, v2, v3 = X[f[0]], X[f[1]], X[f[2]]
        dist, worst_pt = _spherical_triangle_worst_point(v1, v2, v3)
        
        # Gradient of distance for this facet
        # If acute: grad = -worst_pt / sin(dist) ?
        # d/dv arccos(dot(w, v)) = -w / sqrt(1 - dot^2) = -w / sin(dist)
        # We want to minimize, so we move in direction -grad = w / sin(dist)
        
        if dist < 1e-10:
            continue
            
        sin_dist = np.sin(dist)
        if sin_dist < 1e-10:
            continue
            
        factor = w * beta / sin_dist
        
        for j, vj in enumerate([v1, v2, v3]):
            # Project worst_pt to tangent plane at vj
            vec = worst_pt - vj
            proj = vec - np.dot(vec, vj) * vj
            grad[f[j]] += factor * proj
    
    return log_sum_exp, grad.flatten()

def _optimize_lgbfgs(config, deadline):
    """
    Optimize using L-BFGS-B with the soft-max objective.
    """
    config = config.copy()
    x0 = config.flatten()
    
    def fun_grad(x):
        val, grad = _soft_max_objective(x)
        return val, grad
    
    best_x = x0
    best_val, _ = fun_grad(x0)
    
    # Run L-BFGS
    while time.time() < deadline - 10:
        res = minimize(fun_grad, best_x, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        else:
            # If it fails, try with a lower beta or break
            break
            
    best_cfg = _normalize_batch(best_x.reshape(N, 3))
    return best_cfg

def _local_polish(config, deadline):
    """
    Local polish: move points in the worst facet towards the worst point.
    """
    config = config.copy()
    best_config = config.copy()
    
    # Calculate initial score
    try:
        hull = ConvexHull(config)
        facets = hull.simplices
        max_r = 0
        for f in facets:
            v1, v2, v3 = config[f[0]], config[f[1]], config[f[2]]
            d, _ = _spherical_triangle_worst_point(v1, v2, v3)
            if d > max_r: max_r = d
        best_score = max_r
    except:
        return config

    step_size = 0.05
    
    while time.time() < deadline - 5:
        # Find worst facet
        hull = ConvexHull(config)
        facets = hull.simplices
        max_r = 0
        worst_facet = None
        worst_pt = None
        
        for f in facets:
            v1, v2, v3 = config[f[0]], config[f[1]], config[f[2]]
            d, wp = _spherical_triangle_worst_point(v1, v2, v3)
            if d > max_r:
                max_r = d
                worst_facet = f
                worst_pt = wp
        
        if worst_facet is None:
            break
            
        # Try to improve by moving vertices of worst_facet towards worst_pt
        improved = False
        for idx in worst_facet:
            v = config[idx]
            vec = worst_pt - v
            proj = vec - np.dot(vec, v) * v
            norm = np.linalg.norm(proj)
            if norm < 1e-10:
                continue
            
            dir_vec = proj / norm
            
            for ss in [step_size, step_size * 0.5, step_size * 0.1]:
                new_v = v + ss * dir_vec
                new_v = _normalize(new_v)
                
                old_v = config[idx].copy()
                config[idx] = new_v
                
                # Check new score
                try:
                    hull_new = ConvexHull(config)
                    facets_new = hull_new.simplices
                    new_r = 0
                    for fn in facets_new:
                        v1, v2, v3 = config[fn[0]], config[fn[1]], config[fn[2]]
                        d, _ = _spherical_triangle_worst_point(v1, v2, v3)
                        if d > new_r: new_r = d
                    
                    if new_r < best_score - 1e-10:
                        best_score = new_r
                        best_config = config.copy()
                        improved = True
                        break
                    else:
                        config[idx] = old_v
                except:
                    config[idx] = old_v
                    break
            
            if improved:
                break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-4:
                break
                
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Initialization
    config = _fib_sample(N)
    
    # 2. L-BFGS Optimization
    if time.time() < deadline - 30:
        config = _optimize_lgbfgs(config, deadline)
        
    # 3. Local Polish
    if time.time() < deadline - 10:
        config = _local_polish(config, deadline)
        
    # 4. Final Normalization
    config = _normalize_batch(config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
