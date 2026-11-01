import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
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
    
    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0]*a + m[1]*b + m[2]*c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue
    
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

def _kmeans_objective_and_grad(points, target_points):
    """
    Compute the K-Means objective (sum of squared distances to nearest point)
    and its gradient.
    
    Objective: J(p) = sum_{y in Y} min_{i} ||p_i - y||^2
    Gradient: grad_{p_i} J = sum_{y in V_i} 2(p_i - y) = 2 * |V_i| * (p_i - c_i)
    where c_i is the centroid of cluster i.
    
    We return J/N and grad/N to keep scales reasonable.
    """
    # Compute distances from all targets to all points
    # target_points: (M, 3), points: (N, 3)
    # Distances squared: (M, N)
    # ||p_i - y||^2 = ||p_i||^2 + ||y||^2 - 2 p_i . y = 2 - 2 p_i . y (since on sphere)
    dots = np.dot(target_points, points.T) # (M, N)
    dist_sq = 2.0 - 2.0 * dots
    
    # Find nearest point for each target
    nearest_idx = np.argmin(dist_sq, axis=1) # (M,)
    
    # Compute objective
    min_dist_sq = np.min(dist_sq, axis=1)
    obj = np.sum(min_dist_sq) / len(target_points)
    
    # Compute gradients
    # grad_{p_i} = sum_{y in V_i} 2(p_i - y)
    # We accumulate sums of y for each i
    sums = np.zeros_like(points)
    counts = np.zeros(N)
    
    np.add.at(sums, nearest_idx, target_points)
    np.add.at(counts, nearest_idx, 1)
    
    grads = np.zeros_like(points)
    for i in range(N):
        if counts[i] > 0:
            centroid = sums[i] / counts[i]
            # Gradient is 2 * (p_i - centroid)
            # But we are minimizing sum ||p_i - y||^2, so grad is 2(p_i - centroid)
            # We want to move p_i towards centroid, so direction is -(p_i - centroid) = centroid - p_i
            # Wait, minimize J -> grad points in direction of increase.
            # L-BFGS minimizes, so we pass grad.
            # grad = 2(p_i - centroid).
            # If we move p_i by -alpha * grad, we move towards centroid.
            grads[i] = 2.0 * (points[i] - centroid)
        else:
            # If a point has no targets, it's isolated. 
            # This shouldn't happen with good initialization, but if it does, 
            # we should probably move it towards the center of the largest empty region?
            # For now, set gradient to 0 or small random? 
            # Better: if count is 0, it's not contributing to loss, but it's "wasted".
            # We can add a penalty for unused points or just let it stay.
            # Actually, if a point has 0 targets, it means all targets are closer to other points.
            # This is a local minimum of the k-means objective (isolated cluster).
            # To break symmetry, we might need a repulsion term, but let's assume 
            # the fine target grid ensures all points get some targets.
            pass
    
    # Project gradients to tangent space
    for i in range(N):
        radial = np.dot(grads[i], points[i]) * points[i]
        grads[i] = grads[i] - radial
    
    return obj, grads

def _lbfgs_stage(points, time_budget, target_points):
    """
    Use L-BFGS-B to minimize the K-Means objective (CVT).
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        
        obj, grads = _kmeans_objective_and_grad(pts, target_points)
        return obj, grads.reshape(-1)
    
    x0 = points.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100}
        )
        if result.success:
            best_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            best_pts = best_pts / norms
            return best_pts
    except Exception:
        pass
    
    return points

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    Directly minimizes the soft-max covering radius.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-12
    decay = 0.85
    growth = 1.08
    
    while time.time() - start_time < time_budget:
        # Re-use the soft-max logic from the original code for the final polish
        # We need to re-implement or keep the original soft-max functions
        # Since we are replacing the main driver, we keep the original polish functions 
        # but we need the helper functions for them.
        
        # Let's just use a simple gradient step on the max radius for polish
        # Or better, keep the original _soft_max_radius_and_grad
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
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        new_points = new_points / norms
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.15)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
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
        
        grad_contrib = -C / sin_r
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_radius = np.inf
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    # Create a fine target grid for K-Means objective
    # Use a Fibonacci sphere with 2000 points
    M = 2000
    j_t = np.arange(M) + 0.5
    z_t = 1.0 - 2.0 * j_t / M
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j_t
    r_t = np.sqrt(1.0 - z_t * z_t)
    target_points = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    n_seeds = len(seeds)
    
    # Stage 1: L-BFGS optimization of K-Means objective (CVT) for each seed
    base_time_per_seed = 10.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 20:
            break
        
        if i < n_seeds - 1:
            time_budget = min(base_time_per_seed, remaining / (n_seeds - i))
        else:
            time_budget = min(20.0, remaining)
        
        if time_budget < 5.0:
            break
        
        # Apply L-BFGS stage with K-Means objective
        pts = _lbfgs_stage(seed_pts, time_budget, target_points)
        
        # Quick local polish on covering radius
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            pts, rad = _local_polish(pts, min(8.0, remaining - 10), beta=80.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Fine local polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 20:
            budget = min(25.0, remaining - 12)
            if budget > 5:
                pts, rad = _local_polish(best_points, budget, beta=300.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with higher beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 12:
            budget = min(15.0, remaining - 8)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=800.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
