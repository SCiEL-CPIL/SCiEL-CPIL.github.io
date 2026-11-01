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

def _repulsive_forces(points, exponent=2.0, min_dist=0.05):
    """
    Compute repulsive forces between all pairs of points.
    Force magnitude ~ 1 / d^exponent, direction away from other point.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i+1, n):
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d < min_dist:
                d = min_dist
            # Tangential component only (project onto tangent plane at point i)
            d_hat = d_vec / d
            # Tangential component at point i
            radial_i = np.dot(d_hat, points[i]) * points[i]
            tangential_i = d_hat - radial_i
            mag = 1.0 / (d ** exponent)
            forces[i] += mag * tangential_i
            forces[j] -= mag * tangential_i
    
    return forces

def _pressure_towards_worst(points, centers, radii, pressure_strength=1.0):
    """
    Compute pressure forces pushing points away from worst facet circumcenters.
    Points on the worst facet get pushed away from the circumcenter.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    if len(centers) == 0:
        return forces
    
    # Find top few worst facets
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
        
        # Weight by how bad this facet is
        weight = radii[idx] ** 2 / (np.sum(radii**2) + 1e-10)
        
        for i in range(n):
            # Distance from point i to circumcenter C
            dot_ic = np.dot(points[i], C)
            if abs(dot_ic) > 1.0:
                continue
            # Only push points that are "near" this circumcenter
            ang_dist = np.arccos(np.clip(dot_ic, -1.0, 1.0))
            if ang_dist > r + 0.1:
                continue
            
            # Force direction: away from C, tangential to sphere at point i
            radial_C = dot_ic * points[i]
            tangential_C = C - radial_C
            tangential_C_norm = np.linalg.norm(tangential_C)
            if tangential_C_norm < 1e-10:
                continue
            tangential_C = tangential_C / tangential_C_norm
            
            # Strength decreases with angular distance from circumcenter
            strength = pressure_strength * weight * np.cos(ang_dist / r * np.pi / 2)
            if strength > 0:
                forces[i] += strength * tangential_C
    
    return forces

def _combined_forces(points, beta_pressure=1.0, repulsive_strength=0.5):
    """
    Combine repulsive forces and pressure forces.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    
    if hull is None:
        return None, np.pi
    
    repulsive = _repulsive_forces(points, exponent=2.0, min_dist=0.03)
    pressure = _pressure_towards_worst(points, centers, radii, pressure_strength=beta_pressure)
    
    total = repulsive_strength * repulsive + pressure
    
    # Normalize to unit sphere after computing forces
    current_radius = radii[worst_idx] if worst_idx is not None else np.pi
    
    return total, current_radius

def _local_search_repulsive(points, time_budget, beta_pressure=1.0, repulsive_strength=0.5):
    """
    Local search using repulsive + pressure forces with adaptive step size.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-7
    decay = 0.95
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        forces, current_radius = _combined_forces(current_points, beta_pressure, repulsive_strength)
        
        if forces is None:
            break
        
        if current_radius < best_radius - 1e-12:
            best_radius = current_radius
            best_points = current_points.copy()
        
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-12:
            break
        
        # Normalize forces and apply step
        direction = forces / force_norm
        new_points = current_points + step_size * direction
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < current_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _local_search_gradient(points, time_budget, beta=100.0):
    """
    Gradient-based local search using soft-max of facet radii.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-7
    decay = 0.95
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(current_points)
        if hull is None or worst_idx is None:
            break
        
        # Soft-max radius and gradients
        max_r = np.max(radii)
        exp_terms = np.exp(beta * (radii - max_r))
        log_sum = np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(current_points)
        for i in range(len(hull.simplices)):
            if radii[i] < 1e-10 or weights[i] < 1e-12:
                continue
            w = weights[i]
            C = centers[i]
            r = radii[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            grad_contrib = -C / sin_r
            simplex = hull.simplices[i]
            for k in range(3):
                p_idx = simplex[k]
                grads[p_idx] += w * grad_contrib
        
        if soft_radius < best_radius - 1e-12:
            best_radius = soft_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-12:
            break
        
        grads = grads / grad_norm
        new_points = current_points + step_size * grads
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < soft_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _simulated_annealing(points, time_budget, T_start=0.05, T_end=0.001):
    """
    Simulated annealing on the sphere using random perturbations.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    total_time = time_budget
    
    while time.time() - start_time < time_budget:
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        rng = np.random.default_rng()
        noise = rng.normal(size=(N, 3)) * T
        new_points = current_points + noise
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        delta = new_radius - best_radius
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_points = new_points
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
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
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based seed (5 vertices, subdivide)
    # Start with icosahedron and subdivide faces
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    seeds.append(ico_verts[:50] if len(ico_verts) >= 50 else np.vstack([ico_verts, rng.normal(size=(50-len(ico_verts), 3))]))
    
    # Process seeds with different optimization methods
    n_seeds = len(seeds)
    base_time_per_seed = 10.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 15:
            break
        
        if i < n_seeds - 1:
            time_budget = min(base_time_per_seed, remaining / (n_seeds - i))
        else:
            time_budget = min(25.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Alternate between repulsive and gradient methods
        if i % 2 == 0:
            pts, rad = _local_search_repulsive(seed_pts, time_budget=time_budget, beta_pressure=1.0, repulsive_strength=0.5)
        else:
            pts, rad = _local_search_gradient(seed_pts, time_budget=time_budget, beta=100.0)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Final polish with simulated annealing
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 10:
            budget = min(15.0, remaining - 3)
            if budget > 3:
                pts, rad = _simulated_annealing(best_points, time_budget=budget, T_start=0.03, T_end=0.0005)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Final high-precision polish
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 5:
            budget = min(10.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search_repulsive(best_points, time_budget=budget, beta_pressure=1.5, repulsive_strength=0.3)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
