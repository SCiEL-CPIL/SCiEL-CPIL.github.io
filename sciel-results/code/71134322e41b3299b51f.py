import numpy as np
import scipy.spatial
import time

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
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

def _compute_soft_max_radius_and_grads(points, beta=50.0):
    """
    Compute soft-max of facet radii and analytical gradients.
    This provides a smooth objective for optimization.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Numerical stability: shift by max
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft radius w.r.t. points
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
        
        # d(r_i)/d(point_k) for k in simplex i
        # r_i = arccos(dot(C_i, a_k))
        # dr/dp = -C_i / sin(r_i) projected onto tangent plane
        # But C_i depends on the points too. For a first-order approximation,
        # we treat C_i as fixed (which is valid for small steps).
        # The gradient of arccos(x) is -1/sqrt(1-x^2) = -1/sin(r)
        # So d r_i / d p_k = -C_i / sin(r_i)
        
        grad_contrib = -C / sin_r
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            # Project gradient onto tangent plane at point p_idx
            p = points[p_idx]
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _optimize_points(points, time_budget, beta=50.0, method='gradient'):
    """
    Optimize points using gradient descent on soft-max covering radius.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.05
    min_step = 1e-8
    decay = 0.92
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _compute_soft_max_radius_and_grads(current_points, beta=beta)
        
        # Also track the true radius
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-12:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-14:
            break
        
        # Normalize gradient
        direction = grads / grad_norm
        
        # Try a step
        new_points = current_points + step_size * direction
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_true_radius = _compute_covering_radius(new_points)
        
        if new_true_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.2)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _repulsive_optimize(points, time_budget, repulsive_strength=1.0, pressure_strength=2.0):
    """
    Optimize using a combination of repulsive forces and pressure from worst facets.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-8
    decay = 0.9
    growth = 1.08
    
    while time.time() - start_time < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(current_points)
        if hull is None:
            break
        
        true_radius = radii[worst_idx] if worst_idx is not None else np.pi
        if true_radius < best_radius - 1e-12:
            best_radius = true_radius
            best_points = current_points.copy()
        
        # Compute repulsive forces
        n = len(current_points)
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                d_vec = current_points[i] - current_points[j]
                d = np.linalg.norm(d_vec)
                if d < 0.01:
                    d = 0.01
                d_hat = d_vec / d
                radial_i = np.dot(d_hat, current_points[i]) * current_points[i]
                tangential_i = d_hat - radial_i
                mag = 1.0 / (d ** 2)
                forces[i] += repulsive_strength * mag * tangential_i
                forces[j] -= repulsive_strength * mag * tangential_i
        
        # Pressure from worst facets
        if worst_idx is not None and len(centers) > 0:
            sorted_idx = np.argsort(radii)[::-1]
            n_worst = min(5, len(sorted_idx))
            
            for idx in sorted_idx[:n_worst]:
                if radii[idx] < 1e-10:
                    continue
                C = centers[idx]
                r = radii[idx]
                sin_r = np.sin(r)
                if sin_r < 1e-10:
                    continue
                
                weight = (radii[idx] / np.max(radii)) ** 2
                
                for i in range(n):
                    dot_ic = np.dot(current_points[i], C)
                    ang_dist = np.arccos(np.clip(dot_ic, -1.0, 1.0))
                    if ang_dist > r + 0.15:
                        continue
                    
                    radial_C = dot_ic * current_points[i]
                    tangential_C = C - radial_C
                    tangential_C_norm = np.linalg.norm(tangential_C)
                    if tangential_C_norm < 1e-10:
                        continue
                    tangential_C = tangential_C / tangential_C_norm
                    
                    strength = pressure_strength * weight * max(0, 1 - ang_dist / (r + 0.15))
                    forces[i] += strength * tangential_C
        
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-14:
            break
        
        direction = forces / force_norm
        new_points = current_points + step_size * direction
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.15)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

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
    
    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777]:
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
    # Extend icosahedron to 50 points by adding random points
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    n_seeds = len(seeds)
    base_time_per_seed = 8.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 12:
            break
        
        if i < n_seeds - 1:
            time_budget = min(base_time_per_seed, remaining / (n_seeds - i))
        else:
            time_budget = min(20.0, remaining)
        
        if time_budget < 2.0:
            break
        
        # Alternate between gradient and repulsive methods
        if i % 3 == 0:
            pts, rad = _optimize_points(seed_pts, time_budget=time_budget, beta=50.0, method='gradient')
        elif i % 3 == 1:
            pts, rad = _repulsive_optimize(seed_pts, time_budget=time_budget, repulsive_strength=1.0, pressure_strength=2.0)
        else:
            pts, rad = _optimize_points(seed_pts, time_budget=time_budget, beta=100.0, method='gradient')
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Final polish stages
    if best_points is not None:
        # Stage 1: Gradient polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            budget = min(20.0, remaining - 5)
            if budget > 3:
                pts, rad = _optimize_points(best_points, time_budget=budget, beta=100.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Stage 2: Repulsive polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(15.0, remaining - 3)
            if budget > 2:
                pts, rad = _repulsive_optimize(best_points, time_budget=budget, repulsive_strength=0.5, pressure_strength=3.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Stage 3: Fine gradient polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(10.0, remaining - 2)
            if budget > 1:
                pts, rad = _optimize_points(best_points, time_budget=budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
