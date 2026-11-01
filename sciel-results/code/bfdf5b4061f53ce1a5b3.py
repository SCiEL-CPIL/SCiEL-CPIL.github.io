import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
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

def _vectorized_repulsive_force(points, alpha=2.0):
    """
    Compute vectorized pairwise repulsive forces.
    Force between i and j: (1/d^alpha) * tangential component of (p_i - p_j)
    """
    n = len(points)
    # Compute all pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero
    dist_safe = np.maximum(dist, 1e-8)
    
    # Compute force magnitudes: 1/d^alpha, zero out self-interactions
    mag = np.zeros_like(dist)
    mask = dist > 1e-8
    mag[mask] = 1.0 / (dist_safe[mask] ** alpha)
    
    # Force direction: normalize diff, project to tangent space
    diff_norm = diff / dist_safe[:, :, np.newaxis]
    
    # Tangential projection: for each pair (i,j), project diff_norm onto tangent at p_i
    # Tangent at p_i: remove component along p_i
    # radial component of diff_norm[i,j] along p_i
    radial = np.einsum('ij,ik->ij', diff_norm, points[np.newaxis, :, :])  # (n, n)
    tangential = diff_norm - radial[:, :, np.newaxis]  # (n, n, 3)
    
    # Apply magnitude
    forces = np.einsum('ij,ijk->ik', mag, tangential)
    
    return forces

def _facet_pressure_force(points, centers, radii, top_k=10, spread=0.2):
    """
    Compute pressure force pushing points away from worst facet circumcenters.
    """
    n = len(points)
    forces = np.zeros((n, 3))
    
    if len(centers) == 0:
        return forces
    
    # Sort by radius descending, take top_k
    sorted_idx = np.argsort(radii)[::-1][:top_k]
    max_r = radii[sorted_idx[0]] if len(sorted_idx) > 0 else 1.0
    
    for idx in sorted_idx:
        if radii[idx] < 1e-10:
            continue
        C = centers[idx]
        r = radii[idx]
        
        # Weight based on how bad this facet is
        weight = (radii[idx] / max_r) ** 2
        
        for i in range(n):
            p = points[i]
            dot_pc = np.dot(p, C)
            ang_dist = np.arccos(np.clip(dot_pc, -1.0, 1.0))
            
            # Only push points that are within some distance of the facet center
            if ang_dist > r + spread:
                continue
            
            # Tangential direction from p towards C
            radial_C = dot_pc * p
            tangential_C = C - radial_C
            tangential_C_norm = np.linalg.norm(tangential_C)
            if tangential_C_norm < 1e-10:
                continue
            tangential_C = tangential_C / tangential_C_norm
            
            # Strength decays with distance
            strength = weight * max(0, 1 - ang_dist / (r + spread))
            forces[i] += strength * tangential_C
    
    return forces

def _combined_force(points, alpha=2.0, pressure_weight=1.0, top_k=10):
    """
    Combine repulsive and pressure forces.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None:
        return np.zeros_like(points), None
    
    rep_force = _vectorized_repulsive_force(points, alpha=alpha)
    press_force = _facet_pressure_force(points, centers, radii, top_k=top_k)
    
    total = rep_force + pressure_weight * press_force
    return total, worst_idx

def _local_search_force(points, time_budget, alpha=2.0, pressure_weight=1.0, top_k=10):
    """
    Local search using combined force-based method.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-12
    decay = 0.85
    growth = 1.1
    
    while time.time() - start_time < time_budget:
        forces, worst_idx = _combined_force(current_points, alpha=alpha, 
                                            pressure_weight=pressure_weight, top_k=top_k)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
        
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-14:
            break
        
        direction = forces / force_norm
        new_points = current_points + step_size * direction
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.15)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _soft_max_radius_and_grads(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    for i in range(len(hull.simplices)):
        if radii[i] < 1e-12 or weights[i] < 1e-15:
            continue
        w = weights[i]
        C = centers[i]
        r = radii[i]
        sin_r = np.sin(r)
        if sin_r < 1e-12:
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

def _lbfgs_optimize(points, time_budget, beta=50.0):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = _normalize(x.reshape(N, 3))
        soft_r, grads = _soft_max_radius_and_grads(pts, beta=beta)
        grad_flat = grads.reshape(-1)
        return soft_r, grad_flat
    
    x0 = points.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
        )
        if result.success or result.fun < _compute_covering_radius(points):
            best_pts = _normalize(result.x.reshape(N, 3))
            return best_pts
    except Exception:
        pass
    
    return points

def _fine_polish(points, time_budget, beta=200.0):
    """
    Fine polish using projected gradient descent on tangent space.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.01
    min_step = 1e-12
    decay = 0.85
    growth = 1.08
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grads(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-14:
            break
        
        direction = grads / grad_norm
        new_points = current_points + step_size * direction
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.05)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
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
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415, 271, 888, 10000]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra = _normalize(extra)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    n_seeds = len(seeds)
    
    # Stage 1: For each seed, do L-BFGS then force-based local search
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 15:
            break
        
        # Time allocation
        if i < n_seeds - 1:
            lbfgs_budget = min(5.0, remaining / (n_seeds - i) * 0.5)
            force_budget = min(5.0, remaining / (n_seeds - i) * 0.5)
        else:
            lbfgs_budget = min(8.0, remaining * 0.4)
            force_budget = min(8.0, remaining * 0.4)
        
        if lbfgs_budget < 2.0:
            break
        
        # L-BFGS stage
        pts = _lbfgs_optimize(seed_pts, lbfgs_budget, beta=50.0)
        
        # Force-based local search
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10 and force_budget > 2.0:
            pts, rad = _local_search_force(pts, min(force_budget, remaining - 5), 
                                           alpha=2.0, pressure_weight=1.0, top_k=10)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Repulsive/pressure force polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 25:
            budget = min(20.0, remaining - 12)
            if budget > 5:
                pts, rad = _local_search_force(best_points, budget, 
                                               alpha=2.0, pressure_weight=1.5, top_k=15)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Fine gradient polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 18:
            budget = min(15.0, remaining - 10)
            if budget > 3:
                pts, rad = _fine_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(8.0, remaining - 5)
            if budget > 2:
                pts, rad = _fine_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    best_points = _normalize(best_points)
    
    return best_points
