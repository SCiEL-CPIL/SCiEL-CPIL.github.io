import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    
    # Initialize outputs
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
    if not np.all(valid):
        if np.any(valid):
            a_v = a[valid]
            b_v = b[valid]
            c_v = c[valid]
            n_v = n[valid]
            norms_v = norms[valid]
            
            u = n_v / norms_v[:, np.newaxis]
            centroids = (a_v + b_v + c_v) / 3.0
            dots = np.sum(u * centroids, axis=1)
            u[dots < 0] *= -1
            
            cos_theta = np.sum(u * a_v, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii[valid] = np.arccos(cos_theta)
            centers[valid] = u
            
        return radii, centers, simplices

    # All valid
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def get_covering_radius(points):
    """
    Compute the covering radius (max facet circumradius) in radians.
    """
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: Farthest Point Sampling.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(2000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
        
    return np.array(points)

def lloyd_step(points, alpha=0.5):
    """
    Perform one step of Lloyd's algorithm.
    Move each point towards the area-weighted centroid of its Voronoi cell.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points

    radii, centers, simplices = compute_facet_data(points)
    
    if len(radii) == 0:
        return points

    updates = np.zeros_like(points)
    counts = np.zeros(N)
    
    for m in range(len(centers)):
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        # Weight by facet area (proportional to sin(r))
        weight = np.sin(radii[m])
        updates[i_idx] += weight * centers[m]
        updates[j_idx] += weight * centers[m]
        updates[k_idx] += weight * centers[m]
        counts[i_idx] += weight
        counts[j_idx] += weight
        counts[k_idx] += weight
    
    valid = counts > 1e-10
    updates[valid] /= counts[valid, np.newaxis]
    
    new_points = points.copy()
    new_points[valid] = points[valid] + alpha * (updates[valid] - points[valid])
    
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def facet_push_step(points, radii, centers, simplices, T=0.1, percentile=80):
    """
    Push vertices of large facets towards the circumcenter.
    """
    if len(radii) == 0:
        return points
        
    threshold = np.percentile(radii, percentile)
    mask = radii > threshold
    
    if not np.any(mask):
        max_idx = np.argmax(radii)
        mask = np.zeros(len(radii), dtype=bool)
        mask[max_idx] = True
        
    updates = np.zeros_like(points)
    
    for m in np.where(mask)[0]:
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        center = centers[m]
        radius = radii[m]
        
        step = T * radius
        
        updates[i_idx] += step * (center - points[i_idx])
        updates[j_idx] += step * (center - points[j_idx])
        updates[k_idx] += step * (center - points[k_idx])
        
    new_points = points + updates
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def smooth_objective_and_grad(points):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    Uses log-sum-exp of squared radii.
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Adaptive beta
    r_max = np.max(radii)
    r_min = np.min(radii)
    spread = r_max - r_min
    beta = max(50.0, min(200.0, 50.0 / max(spread, 0.01)))
    
    max_r2 = np.max(radii**2)
    scaled = beta * (radii**2 - max_r2)
    
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    log_sum_exp_val = beta * max_r2 + max_scaled + np.log(sum_exp)
    
    value = np.sqrt(log_sum_exp_val / beta)
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        coeff = weights[m] * 2.0 * beta * r / (sin_r * sin_r)
        
        for idx in [i_idx, j_idx, k_idx]:
            p = points[idx]
            tangent_center = center - np.dot(center, p) * p
            grad_points[idx] += coeff * tangent_center
        
    gnorm = np.linalg.norm(grad_points)
    if gnorm > 1e-10:
        grad_points /= gnorm
        
    return value, grad_points

def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass with adaptive step sizes and exact local optimization.
    """
    start = time.time()
    alpha = start_alpha
    best_local = points.copy()
    best_local_radius = np.pi
    
    # Phase 1: Gradient descent on smooth surrogate
    for step in range(1000):
        if time.time() - start > time_budget * 0.4:
            break
            
        val, grad = smooth_objective_and_grad(points)
        
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
                
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        points = points + alpha * grad_tangent
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha *= 0.98
        if alpha < 1e-5:
            break
            
    points = best_local
    
    # Phase 2: Facet push with adaptive percentile
    alpha_push = 0.05
    for step in range(2000):
        if time.time() - start > time_budget * 0.7:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
            
        pct = max(50, 95 - step * 0.05)
        threshold = np.percentile(radii, pct)
        mask = radii > threshold
        
        if not np.any(mask):
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
            
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            idx = simplices[m]
            i_idx, j_idx, k_idx = idx
            center = centers[m]
            r = radii[m]
            
            step_size = alpha_push * r
            
            updates[i_idx] += step_size * (center - points[i_idx])
            updates[j_idx] += step_size * (center - points[j_idx])
            updates[k_idx] += step_size * (center - points[k_idx])
            
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha_push *= 0.99
        if alpha_push < 1e-5:
            break
            
    points = best_local
    
    # Phase 3: Exact local optimization - coordinate descent on max facet
    for step in range(500):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
            
        max_idx = np.argmax(radii)
        if radii[max_idx] < 0.01:
            break
            
        i_idx, j_idx, k_idx = simplices[max_idx]
        center = centers[max_idx]
        
        best_improvement = 0
        best_new_points = None
        
        for v_idx in [i_idx, j_idx, k_idx]:
            for step_frac in [0.01, 0.05, 0.1, 0.2, 0.3]:
                new_points = points.copy()
                new_points[v_idx] = points[v_idx] + step_frac * (center - points[v_idx])
                new_points[v_idx] /= np.linalg.norm(new_points[v_idx])
                
                new_radii, _, _ = compute_facet_data(new_points)
                if len(new_radii) == 0:
                    continue
                    
                new_max_r = np.max(new_radii)
                improvement = max_r - new_max_r
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_new_points = new_points
        
        if best_new_points is not None and best_improvement > 1e-8:
            points = best_new_points
        else:
            np.random.seed(step)
            perturb = np.random.randn(N, 3) * 0.01
            perturb -= np.sum(perturb * points, axis=1, keepdims=True) * points
            new_points = points + perturb
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_radii, _, _ = compute_facet_data(new_points)
            if len(new_radii) > 0:
                new_max_r = np.max(new_radii)
                if new_max_r < max_r:
                    points = new_points
                else:
                    break
    
    return best_local

def spectral_refinement(points, time_budget):
    """
    Structural change: Spherical Linear Programming (SLP) dual-inspired refinement.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    n_grid = 2000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    sigma = 0.1
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        
        dots = grid_points @ points.T
        dist_sq = 2.0 - 2.0 * dots
        dist_sq = np.maximum(dist_sq, 0.0)
        
        weights = np.exp(-dist_sq / (2.0 * sigma**2))
        potential = np.sum(weights, axis=1)
        
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
        
        threshold = np.percentile(potential, 20)
        low_mask = potential < threshold
        
        if not np.any(low_mask):
            break
            
        low_points = grid_points[low_mask]
        
        updates = np.zeros_like(points)
        
        for i in range(N):
            dists = np.linalg.norm(low_points - points[i], axis=1)
            nearest_idx = np.argmin(dists)
            target = low_points[nearest_idx]
            
            step_size = 0.01
            updates[i] = step_size * (target - points[i])
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        sigma *= 0.99
        
        if sigma < 0.01:
            break
    
    return best_local

def nelder_mead_refine(points, time_budget, max_iter=2000):
    """
    Nelder-Mead simplex method refinement on the sphere.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = np.pi
    
    radii, _, _ = compute_facet_data(points)
    if len(radii) > 0:
        best_radius = np.max(radii)
    
    np.random.seed(42)
    current = points.copy()
    no_improve = 0
    
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        base_scale = 0.05 * (0.95 ** (i / 30.0))
        if no_improve > 20:
            base_scale *= 0.5
        scale = max(0.001, base_scale)
            
        perturbation = np.random.randn(N, 3) * scale
        perturbation -= np.sum(perturbation * current, axis=1, keepdims=True) * current
        new_points = current + perturbation
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        radii, _, _ = compute_facet_data(new_points)
        if len(radii) == 0:
            continue
            
        new_radius = np.max(radii)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            current = new_points.copy()
            no_improve = 0
        elif new_radius < best_radius * 1.005:
            if np.random.rand() < 0.2:
                current = new_points.copy()
            no_improve += 1
        else:
            if np.random.rand() < 0.05:
                current = new_points.copy()
            no_improve += 1
    
    return best_points

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(init_fib)
    
    # 2. Greedy Initialization
    init_greedy = greedy_init(N, seed=10)
    initializations.append(init_greedy)
    
    # 3. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 4. Greedy 2
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 6. Greedy 3
    init_greedy3 = greedy_init(N, seed=999)
    initializations.append(init_greedy3)
    
    # 7. Fibonacci with different phase
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.3)
    r2 = np.sqrt(1.0 - z2 * z2)
    init_fib2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    initializations.append(init_fib2)
    
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 30.0:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            if best_points is not None:
                np.random.seed(restart * 137)
                points = best_points + np.random.randn(N, 3) * 0.05
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        T = 0.05
        alpha_lloyd = 0.5
        no_improve_count = 0
        current_radius = np.pi
        
        while True:
            if time.time() - start_time > time_limit - 30.0:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    T *= 0.9
                    alpha_lloyd *= 0.95
                if no_improve_count > 100:
                    break
                if T < 1e-6 and alpha_lloyd < 1e-6:
                    break
            
            if no_improve_count % 10 == 0:
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                points = facet_push_step(points, radii, centers, simplices, T=T, percentile=80)
                
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            T *= 0.995
            alpha_lloyd *= 0.995
            
        if time.time() - start_time < time_limit - 25.0:
            nm_time = min(8.0, time_limit - 25.0 - (time.time() - start_time))
            refined = nelder_mead_refine(points, nm_time)
            radii, _, _ = compute_facet_data(refined)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = refined.copy()
                    
        if time.time() - start_time < time_limit - 15.0:
            polished = spectral_refinement(points, time_budget=min(8.0, time_limit - 15.0 - (time.time() - start_time)))
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    
    if best_points is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5.0:
            polished = polish(best_points, remaining_time)
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    else:
        best_points = init_fib
        
    return best_points
