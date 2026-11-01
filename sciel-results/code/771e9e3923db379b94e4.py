import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 145.0  # safety margin from 150s

def _normalize(points):
    """Normalize points to unit sphere."""
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
        hull = ConvexHull(points)
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

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Uses a more numerically stable approach.
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

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _repulsive_relaxation(points, time_budget, iterations=200):
    """
    Vectorized repulsion-based relaxation to improve initial spread.
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    step = 0.005
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Compute pairwise repulsive forces vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists_safe = np.where(dists > 1e-10, dists, 1e-10)
        inv_dist3 = 1.0 / (dists_safe ** 3)
        np.fill_diagonal(inv_dist3, 0.0)
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        # Project to tangent space
        radial_comps = np.sum(forces * pts, axis=1, keepdims=True)
        tangential_forces = forces - radial_comps * pts
        
        # Normalize forces
        force_norms = np.linalg.norm(tangential_forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        normalized_forces = tangential_forces / force_norms
        
        new_pts = pts + step * normalized_forces
        new_pts = _project_to_sphere(new_pts)
        
        new_radius = _compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step *= 0.8
            if step < 1e-6:
                break
    
    return best_pts, best_radius

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    Uses adaptive beta: starts moderate, increases for finer convergence.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_soft_r = _compute_covering_radius(points)
    
    for beta in [50.0, 120.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=beta)
            grad_flat = grads.reshape(-1)
            return soft_r, grad_flat
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 600, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.fun < best_soft_r - 1e-10:
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
                best_soft_r = _compute_covering_radius(best_pts)
        except Exception:
            pass
    
    return best_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    Directly minimizes the soft-max covering radius.
    Uses adaptive step size with line search.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-13
    decay = 0.8
    growth = 1.1
    stagnant = 0
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
            stagnant = 0
        else:
            stagnant += 1
        
        if stagnant > 50:
            break
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-15:
            break
        
        direction = -grads / grad_norm
        new_points = current_points + step_size * direction
        new_points = _project_to_sphere(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.12)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _worst_facet_optimization(points, step_size, time_budget, iterations=200):
    """
    Iteratively move the vertices of the worst facet towards the worst center.
    Uses adaptive step size and tangent space projection.
    """
    start_time = time.time()
    pts = points.copy()
    prev_radius = np.pi
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        radius, worst_center = _compute_covering_radius(pts), None
        # Get worst center
        hull, centers, radii, worst_idx = _compute_facet_data(pts)
        if hull is None or worst_idx is None:
            break
        worst_center = centers[worst_idx]
        
        if radius >= prev_radius:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        else:
            prev_radius = radius
        
        # Find facets that are close to the worst center
        target_indices_set = set()
        for i, simplex in enumerate(hull.simplices):
            c = centers[i]
            if np.dot(c, worst_center) > 0.999:
                for idx in simplex:
                    target_indices_set.add(idx)
        
        if not target_indices_set:
            break
        
        # Move vertices towards worst_center in tangent space
        for idx in target_indices_set:
            v = pts[idx]
            dir_vec = worst_center - v
            radial_comp = np.dot(dir_vec, v)
            dir_vec = dir_vec - radial_comp * v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                pts[idx] = v + step_size * dir_vec
                pts[idx] /= np.linalg.norm(pts[idx])
    
    return pts

def _lloyd_iteration(points, time_budget, iterations=50):
    """
    Lloyd's algorithm for spherical k-means as a local refinement step.
    Each point moves to the centroid of its Voronoi cell (approximated by nearest-neighbor regions).
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Assign each point to its nearest neighbor's region (simple approximation)
        # Compute pairwise distances
        dots = pts @ pts.T
        # For each point, find points in its "Voronoi cell" (approximated by nearest neighbors)
        # Use a simple approach: for each point, move towards the centroid of nearby points
        new_pts = np.zeros_like(pts)
        
        for i in range(N):
            # Find points within a certain angular distance
            dists = np.arccos(np.clip(dots[i], -1.0, 1.0))
            # Points within 0.5 radians are considered neighbors
            mask = dists < 0.5
            if np.sum(mask) > 1:
                # Centroid of neighbors (excluding self)
                neighbors = pts[mask]
                centroid = np.sum(neighbors, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    centroid /= centroid_norm
                    # Move towards centroid
                    new_pts[i] = pts[i] * 0.7 + centroid * 0.3
                else:
                    new_pts[i] = pts[i]
            else:
                new_pts[i] = pts[i]
        
        new_pts = _project_to_sphere(new_pts)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            break
    
    return best_pts, best_radius

def _maximize_hull_volume(points, time_budget, iterations=50):
    """
    Move points to maximize the convex hull volume.
    This acts as a 'pressure minimization' or equilibrium search.
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    step_size = 0.01
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        hull, centers, radii, worst_idx = _compute_facet_data(pts)
        if hull is None:
            break
        
        # Compute gradient of volume
        vol_grads = np.zeros_like(pts)
        for simplex in hull.simplices:
            a_idx, b_idx, c_idx = simplex
            a = pts[a_idx]
            b = pts[b_idx]
            c = pts[c_idx]
            
            # Volume contribution is 1/6 * dot(a, cross(b, c))
            # Grad wrt a: 1/6 * cross(b, c)
            # Grad wrt b: 1/6 * cross(c, a)
            # Grad wrt c: 1/6 * cross(a, b)
            
            grad_a = np.cross(b, c) / 6.0
            grad_b = np.cross(c, a) / 6.0
            grad_c = np.cross(a, b) / 6.0
            
            # Project to tangent space
            vol_grads[a_idx] += grad_a - np.dot(grad_a, a) * a
            vol_grads[b_idx] += grad_b - np.dot(grad_b, b) * b
            vol_grads[c_idx] += grad_c - np.dot(grad_c, c) * c
        
        # Normalize gradient
        grad_norms = np.linalg.norm(vol_grads, axis=1, keepdims=True)
        grad_norms = np.maximum(grad_norms, 1e-15)
        normalized_grads = vol_grads / grad_norms
        
        new_pts = pts + step_size * normalized_grads
        new_pts = _project_to_sphere(new_pts)
        
        new_radius = _compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
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
    
    n_seeds = len(seeds)
    
    # Stage 1: For each seed, apply repulsion relaxation, then L-BFGS, then local polish
    stage1_deadline = deadline - 30.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 5:
            break
        
        if i < n_seeds - 1:
            time_budget = min(8.0, remaining / (n_seeds - i))
        else:
            time_budget = min(12.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Step 1: Repulsion relaxation for initial spread
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _repulsive_relaxation(seed_pts, min(3.0, remaining - 2))
        else:
            pts = seed_pts
            rad = _compute_covering_radius(pts)
        
        # Step 2: L-BFGS optimization
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts = _lbfgs_stage(pts, min(4.0, remaining - 2))
        
        # Step 3: Local polish
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _local_polish(pts, min(3.0, remaining - 2), beta=80.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Volume maximization (pressure minimization)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            pts, rad = _maximize_hull_volume(best_points, min(5.0, remaining - 10), iterations=50)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
        
        # Worst-facet optimization with adaptive step sizes
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            for step in [0.05, 0.02, 0.01, 0.005]:
                elapsed = time.time() - start_time
                remaining = deadline - elapsed
                if remaining < 5:
                    break
                pts = _worst_facet_optimization(best_points, step, min(5.0, remaining - 3), iterations=100)
                rad = _compute_covering_radius(pts)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Lloyd's iteration for local refinement
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            pts, rad = _lloyd_iteration(best_points, min(8.0, remaining - 5), iterations=50)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
        
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(10.0, remaining - 5)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta for final convergence
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(8.0, remaining - 4)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(4.0, remaining - 3)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized and valid
    best_points = _normalize(best_points)
    
    # Final sanity check: verify no coincident points, not in hemisphere
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        best_points = seeds[0]
    
    return best_points
