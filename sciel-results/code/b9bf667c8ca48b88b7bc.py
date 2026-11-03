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

def get_max_radius(points):
    """Quickly get the max covering radius."""
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
    Move each point towards the centroid of its Voronoi cell.
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
        updates[i_idx] += centers[m]
        updates[j_idx] += centers[m]
        updates[k_idx] += centers[m]
        counts[i_idx] += 1
        counts[j_idx] += 1
        counts[k_idx] += 1
    
    valid = counts > 0
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
    Uses log-sum-exp of squared radii with proper gradient derivation.
    
    For a facet with vertices a, b, c and outward normal u, the circumradius r satisfies
    cos(r) = u · a. The gradient of r with respect to vertex a is:
    dr/da = -u / sin(r) (projected to tangent space)
    
    For the smooth max via log-sum-exp:
    f = (1/beta) * log(sum(exp(beta * r_m^2)))
    df/da_i = (1/beta) * sum_m w_m * 2*r_m * dr_m/da_i
    where w_m = exp(beta*r_m^2) / sum(exp(beta*r_m^2))
    
    Returns:
        value: float
        grad: array of shape (N, 3)
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    beta = 50.0
    
    # Compute weights for log-sum-exp
    r2 = radii**2
    max_r2 = np.max(r2)
    scaled = beta * (r2 - max_r2)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Value of smooth max
    log_sum_exp_val = max_r2 + np.log(sum_exp) / beta
    value = np.sqrt(log_sum_exp_val)
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-10:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # Gradient of r_m with respect to each vertex: -u/sin(r)
        # Gradient of r_m^2: 2*r_m * (-u/sin(r))
        # Contribution to df/da: (1/beta) * w_m * 2*r_m * (-u/sin(r))
        coeff = (1.0 / beta) * weights[m] * 2.0 * r / sin_r
        
        # The gradient points in the -center direction (push vertex away from circumcenter)
        grad_points[i_idx] += coeff * center
        grad_points[j_idx] += coeff * center
        grad_points[k_idx] += coeff * center
        
    return value, grad_points

def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass using gradient descent on smooth surrogate
    followed by fine facet push.
    """
    start = time.time()
    alpha = start_alpha
    best_local = points.copy()
    best_local_radius = get_max_radius(points)
    
    # Phase 1: Gradient descent on smooth surrogate
    for step in range(2000):
        if time.time() - start > time_budget * 0.4:
            break
            
        val, grad = smooth_objective_and_grad(points)
        
        max_r = get_max_radius(points)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
                
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
            
        points = points + alpha * grad_tangent
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha *= 0.995
        
        if alpha < 1e-6:
            break
            
    points = best_local.copy()
    
    # Phase 2: Fine facet push with decreasing step
    alpha_push = 0.01
    for step in range(3000):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
            
        # Target the top 20% of facets
        threshold = np.percentile(radii, 80)
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
        
        alpha_push *= 0.998
        
        if alpha_push < 1e-6:
            break
            
    # Phase 3: Very fine polish
    alpha_fine = 0.005
    for step in range(2000):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
            
        # Only push the single largest facet
        max_idx = np.argmax(radii)
        idx = simplices[max_idx]
        i_idx, j_idx, k_idx = idx
        center = centers[max_idx]
        r = radii[max_idx]
        
        step_size = alpha_fine * r
        
        updates = np.zeros_like(points)
        updates[i_idx] += step_size * (center - points[i_idx])
        updates[j_idx] += step_size * (center - points[j_idx])
        updates[k_idx] += step_size * (center - points[k_idx])
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha_fine *= 0.999
        
        if alpha_fine < 1e-7:
            break
            
    return best_local

def spectral_refinement(points, time_budget):
    """
    Structural change: Move points towards under-covered regions.
    Uses a grid-based potential to identify poorly covered areas.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = get_max_radius(points)
    
    n_grid = 3000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    sigma = 0.08
    
    for step in range(300):
        if time.time() - start > time_budget:
            break
        
        dots = grid_points @ points.T
        dist_sq = 2.0 - 2.0 * dots
        dist_sq = np.maximum(dist_sq, 0.0)
        
        weights = np.exp(-dist_sq / (2.0 * sigma**2))
        potential = np.sum(weights, axis=1)
        
        max_r = get_max_radius(points)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
        
        # Find the 10% most under-covered grid points
        threshold = np.percentile(potential, 10)
        low_mask = potential < threshold
        
        if not np.any(low_mask):
            break
            
        low_points = grid_points[low_mask]
        
        # For each configuration point, move it slightly towards its nearest under-covered point
        updates = np.zeros_like(points)
        
        # Batch compute distances
        for i in range(N):
            dists = np.linalg.norm(low_points - points[i], axis=1)
            nearest_idx = np.argmin(dists)
            target = low_points[nearest_idx]
            
            step_size = 0.005
            updates[i] = step_size * (target - points[i])
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        sigma *= 0.995
        
        if sigma < 0.02:
            break
    
    return best_local

def nelder_mead_refine(points, time_budget, max_iter=2000):
    """
    Simulated annealing refinement on the sphere.
    Uses temperature-controlled acceptance to escape local minima.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = get_max_radius(points)
    
    np.random.seed(42)
    
    current = points.copy()
    current_radius = best_radius
    
    # Temperature schedule
    T_start = 0.02
    T_end = 0.001
    n_iter = max_iter
    
    for i in range(n_iter):
        if time.time() - start > time_budget:
            break
        
        # Temperature decay
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Generate perturbation with scale proportional to temperature
        scale = T
        perturbation = np.random.randn(N, 3) * scale
        # Project to tangent space
        perturbation -= np.sum(perturbation * current, axis=1, keepdims=True) * current
        new_points = current + perturbation
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = get_max_radius(new_points)
        if np.isnan(new_radius) or np.isinf(new_radius):
            continue
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_radius - current_radius
        if delta < 0:
            current = new_points.copy()
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
        else:
            # Metropolis acceptance
            if T > 1e-8:
                accept_prob = np.exp(-delta / T)
                if np.random.rand() < accept_prob:
                    current = new_points.copy()
                    current_radius = new_radius
    
    return best_points

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin (150s kill, 2s buffer)
    
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
    
    # 8. Greedy 4
    init_greedy4 = greedy_init(N, seed=777)
    initializations.append(init_greedy4)
    
    # 9. Fibonacci with larger noise
    np.random.seed(555)
    init_fib_noise2 = init_fib + np.random.randn(N, 3) * 0.3
    init_fib_noise2 /= np.linalg.norm(init_fib_noise2, axis=1, keepdims=True)
    initializations.append(init_fib_noise2)
    
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        points = initializations[restart].copy()
        
        # Optimization loop: alternate Lloyd and facet push
        T = 0.08
        alpha_lloyd = 0.4
        no_improve_count = 0
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
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
                if no_improve_count > 30:
                    T *= 0.92
                    alpha_lloyd *= 0.95
                if no_improve_count > 80:
                    break
                if T < 1e-6 and alpha_lloyd < 1e-6:
                    break
            
            # Alternate steps
            if no_improve_count % 5 == 0:
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                points = facet_push_step(points, radii, centers, simplices, T=T, percentile=75)
                
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            T *= 0.997
            alpha_lloyd *= 0.997
            
        # Simulated annealing refinement
        elapsed = time.time() - start_time
        if elapsed < time_limit - 20.0:
            sa_time = min(12.0, time_limit - elapsed - 5.0)
            if sa_time > 3.0:
                refined = nelder_mead_refine(points, sa_time)
                max_r = get_max_radius(refined)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = refined.copy()
                    
        # Spectral refinement
        elapsed = time.time() - start_time
        if elapsed < time_limit - 15.0:
            spec_time = min(8.0, time_limit - elapsed - 3.0)
            if spec_time > 2.0:
                polished = spectral_refinement(points, spec_time)
                max_r = get_max_radius(polished)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished.copy()
    
    # Final Polish
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        if remaining_time > 5.0:
            polished = polish(best_points, remaining_time)
            max_r = get_max_radius(polished)
            if max_r < best_radius:
                best_radius = max_r
                best_points = polished
    else:
        best_points = init_fib
        
    return best_points
