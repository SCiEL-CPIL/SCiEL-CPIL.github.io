import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50

def _normalize(points):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return None, None, None, None
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return hull, np.zeros((0,3)), np.zeros(0), None
        
    a = points[facets[:, 0]]  # (F, 3)
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # Vectorized circumcenter computation
    # Normal to the facet
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    # Handle degenerate facets (collinear points)
    valid = norm_n > 1e-12
    
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        
        centers[valid] = cc
        
        # Calculate radii (angular distance from center to vertex a)
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
        
    return hull, centers, radii, worst_idx

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Use log-sum-exp trick for numerical stability
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    n_facets = len(hull.simplices)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-15:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        simplex = hull.simplices[i]
        p_indices = simplex
        p_pts = points[p_indices]
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
        # Projected to tangent space at p: grad_t = grad - (grad . p) p
        grad_contrib = -C / sin_r
        
        dots = np.sum(p_pts * grad_contrib, axis=1) # (3,)
        radial_comps = dots[:, np.newaxis] * p_pts   # (3, 3)
        tangential_grads = grad_contrib - radial_comps # (3, 3)
        
        for k in range(3):
            grads[p_indices[k]] += w * tangential_grads[k]
    
    return soft_radius, grads

def _facet_targeted_move(points, step_size=0.01):
    """
    Perform a targeted move on the worst facets to reduce their circumradius.
    This moves the vertices of the worst facets away from their circumcenters.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return points
    
    # Identify the top few worst facets
    n_worst = min(5, len(radii))
    worst_indices = np.argsort(radii)[-n_worst:]
    
    new_points = points.copy()
    
    for idx in worst_indices:
        r = radii[idx]
        if r < 1e-10:
            continue
        C = centers[idx]
        simplex = hull.simplices[idx]
        
        # Move each vertex in the simplex away from C
        # Direction is tangent to the sphere
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            
            # Vector from p to C
            # We want to move p such that the angle between p and C increases
            # Gradient of angle is -C/sin(r) projected to tangent space
            # So move in direction -grad = C/sin(r) projected
            # Actually, to increase angle, we move in direction of C projected on tangent space
            # Let's just use the gradient of the radius for this specific facet
            # grad_r = -C/sin(r)
            # We want to decrease r, so we move in -grad_r = C/sin(r)
            
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            
            grad = C / sin_r
            # Project to tangent space at p
            radial_comp = np.dot(grad, p) * p
            tangential_grad = grad - radial_comp
            
            new_points[p_idx] = p + step_size * tangential_grad
            
    new_points = _normalize(new_points)
    return new_points

def _lbfgs_stage(points, time_budget, beta_schedule):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    """
    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget:
            break
            
        def objective_and_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        try:
            x0 = current_pts.reshape(-1)
            
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            max_iter = int(50 + 100 * (remaining / time_budget))
            
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            if result.success or result.fun < _compute_covering_radius(current_pts):
                new_pts = _normalize(result.x.reshape(N, 3))
                new_rad = _compute_covering_radius(new_pts)
                
                if new_rad < current_radius:
                    current_pts = new_pts
                    current_radius = new_rad
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
                    
        except Exception:
            pass
            
    return best_pts

def _local_polish(points, time_budget, beta=1000.0):
    """
    Final local polish using projected gradient descent on tangent space.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.01
    min_step = 1e-12
    decay = 0.8
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-15:
            break
        
        direction = grads / grad_norm
        new_points = current_points + step_size * direction
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.05)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _generate_seeds():
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
        
    # 3. Greedy initialization
    def greedy_init(seed_val):
        rng = np.random.default_rng(seed_val)
        pts = [rng.normal(size=3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.normal(size=(5000, 3))
            candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)
        
    seeds.append(greedy_init(101))
    seeds.append(greedy_init(202))
    seeds.append(greedy_init(303))
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_radius = np.inf
    
    seeds = _generate_seeds()
    n_seeds = len(seeds)
    
    # Beta schedule for L-BFGS: Start smooth, get sharper
    beta_schedule = [20.0, 100.0, 500.0]
    
    # Pass 1: Quick optimization for all seeds
    time_per_seed_pass1 = 10.0
    
    results = []
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 30: # Reserve 30s for final polish
            break
            
        budget = min(time_per_seed_pass1, remaining / (n_seeds - i + 1))
        if budget < 2.0:
            break
            
        pts = _lbfgs_stage(seed_pts, budget, beta_schedule)
        rad = _compute_covering_radius(pts)
        results.append((rad, pts, i))
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
            
    # Sort results by radius
    results.sort(key=lambda x: x[0])
    
    # Pass 2: Refine the top 3 candidates with more time and higher beta
    top_candidates = results[:3]
    
    elapsed = time.time() - start_time
    remaining_for_pass2 = deadline - elapsed - 20.0
    
    if remaining_for_pass2 > 10:
        time_per_seed_pass2 = remaining_for_pass2 / len(top_candidates)
        
        for rad, pts, idx in top_candidates:
            if time.time() > deadline - 20:
                break
                
            refine_schedule = [100.0, 500.0, 2000.0]
            
            refined_pts = _lbfgs_stage(pts, time_per_seed_pass2, refine_schedule)
            refined_rad = _compute_covering_radius(refined_pts)
            
            if refined_rad < best_radius:
                best_radius = refined_rad
                best_points = refined_pts.copy()
                
    # Pass 3: Final local polish on the single best
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 5:
            budget = min(18.0, remaining - 2)
            if budget > 1:
                polished_pts, polished_rad = _local_polish(best_points, budget, beta=2000.0)
                if polished_rad < best_radius:
                    best_radius = polished_rad
                    best_points = polished_pts.copy()
                    
    if best_points is None:
        best_points = seeds[0]
        
    # Ensure output is normalized
    best_points = _normalize(best_points)
    
    return best_points
