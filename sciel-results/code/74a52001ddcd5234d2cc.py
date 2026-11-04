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

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: Farthest Point Sampling.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(1000, 3)
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

def compute_smooth_objective_and_grad(points, beta=200.0):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    
    Uses log-sum-exp of radii (not squared) for better conditioning.
    f = (1/beta) * log(sum(exp(beta * r_i)))
    
    For a facet with circumcenter u and radius r, the gradient of r w.r.t.
    vertex a (projected to tangent space at a) is:
        grad_r_a = -(u - cos(r)*a) / sin(r)
    
    This is the projection of -u/sin(r) onto the tangent space.
    
    The gradient of f w.r.t. a is:
        df/da = sum_i w_i * grad_r_i_a
    where w_i = exp(beta*r_i) / sum_j exp(beta*r_j)
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Soft-max: f = (1/beta) * LSE(beta * r_i)
    max_r = np.max(radii)
    scaled = beta * (radii - max_r)
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    log_sum_exp_val = beta * max_r + max_scaled + np.log(sum_exp)
    value = log_sum_exp_val / beta
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-14:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-8:
            continue
            
        cos_r = np.cos(r)
        
        # Gradient of r w.r.t. each vertex (tangent space projection)
        # grad_r_a = -(u - cos_r * a) / sin_r
        # This equals -u/sin_r projected to tangent space at a
        
        for idx in [i_idx, j_idx, k_idx]:
            a = points[idx]
            # Project -center/sin_r onto tangent space at a
            # tangent_grad = (-center/sin_r) - ((-center/sin_r)·a) * a
            # = (-center + cos_r * a) / sin_r
            tangent_grad = (-center + cos_r * a) / sin_r
            grad_points[idx] += weights[m] * tangent_grad
    
    return value, grad_points

def newton_local_solve(points, time_budget, max_iter=500, tol=1e-10):
    """
    Newton-like local solver using KKT conditions for the epigraph formulation.
    
    We minimize t subject to r_i(x) <= t for all facets i.
    At optimum, the active facets (r_i = t) have their gradients summing to zero
    (in tangent space) with positive multipliers.
    
    We use a damped Newton step on the active set.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
        
        # Identify active facets (within tolerance of max)
        tol_active = 1e-4
        active_mask = radii > (max_r - tol_active)
        
        if not np.any(active_mask):
            break
        
        active_indices = np.where(active_mask)[0]
        
        # Compute gradient of the max radius (sum of active facet gradients)
        # At KKT, sum of lambda_i * grad_r_i = 0 in tangent space
        # For simplicity, use equal weights initially
        grad_max = np.zeros_like(points)
        
        for m in active_indices:
            i_idx, j_idx, k_idx = simplices[m]
            center = centers[m]
            r = radii[m]
            sin_r = np.sin(r)
            cos_r = np.cos(r)
            
            if sin_r < 1e-8:
                continue
            
            for idx in [i_idx, j_idx, k_idx]:
                a = points[idx]
                tangent_grad = (-center + cos_r * a) / sin_r
                grad_max[idx] += tangent_grad
        
        # Project gradient to tangent space
        grad_tangent = grad_max - np.sum(grad_max * points, axis=1, keepdims=True) * points
        
        # Newton step: move opposite to gradient
        # Use adaptive step size
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = 0.1
        for ls_iter in range(30):
            new_points = points + alpha * grad_tangent
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_radii, _, _ = compute_facet_data(new_points)
            if len(new_radii) == 0:
                alpha *= 0.5
                continue
            
            new_max_r = np.max(new_radii)
            
            if new_max_r < max_r - 1e-12:
                points = new_points
                break
            else:
                alpha *= 0.5
        
        if alpha < 1e-10:
            break
    
    return best_local

def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass combining gradient descent and Newton solve.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    # Phase 1: Gradient descent with adaptive step
    alpha = start_alpha
    for step in range(2000):
        if time.time() - start > time_budget * 0.4:
            break
            
        val, grad = compute_smooth_objective_and_grad(points, beta=200.0)
        
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
                
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Adaptive step size with backtracking
        alpha_new = alpha
        for ls_iter in range(20):
            new_points = points + alpha_new * grad_tangent
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_radii, _, _ = compute_facet_data(new_points)
            if len(new_radii) == 0:
                alpha_new *= 0.5
                continue
            
            new_max_r = np.max(new_radii)
            
            if new_max_r < max_r - 1e-12:
                points = new_points
                break
            else:
                alpha_new *= 0.5
        
        alpha = max(alpha_new * 1.1, 0.001)
        alpha = min(alpha, 0.1)
        
        if alpha < 1e-6:
            break
    
    points = best_local
    
    # Phase 2: Newton local solve
    remaining = time_budget - (time.time() - start)
    if remaining > 2.0:
        polished = newton_local_solve(points, remaining)
        radii, _, _ = compute_facet_data(polished)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = polished.copy()
    
    return best_local

def nelder_mead_refine(points, time_budget, max_iter=3000):
    """
    Random restart refinement with small perturbations.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = np.pi
    
    radii, _, _ = compute_facet_data(points)
    if len(radii) > 0:
        best_radius = np.max(radii)
    
    current = points.copy()
    np.random.seed(42)
    
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        scale = 0.03 * (0.97 ** (i / 100.0))
        if scale < 0.0005:
            scale = 0.0005
            
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
        elif np.random.rand() < 0.02:
            current = new_points.copy()
                
    return best_points

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin before 150s kill
    
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
    
    max_restarts = 25
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Perturb the best found so far
            if best_points is not None:
                np.random.seed(restart * 137 + 7)
                points = best_points + np.random.randn(N, 3) * 0.03
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # Optimization loop
        T = 0.05
        alpha_lloyd = 0.5
        no_improve_count = 0
        current_radius = np.pi
        
        # Inner loop
        while True:
            if time.time() - start_time > time_limit - 15.0:
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
                    T *= 0.9
                    alpha_lloyd *= 0.95
                if no_improve_count > 80:
                    break
                if T < 1e-6 and alpha_lloyd < 1e-6:
                    break
            
            # Alternate steps
            if no_improve_count % 10 == 0:
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                points = facet_push_step(points, radii, centers, simplices, T=T, percentile=80)
                
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            T *= 0.995
            alpha_lloyd *= 0.995
            
        # Gradient-based refinement
        if time.time() - start_time < time_limit - 20.0:
            grad_time = min(8.0, time_limit - (time.time() - start_time) - 15.0)
            if grad_time > 1.0:
                polished = polish(points, grad_time)
                radii, _, _ = compute_facet_data(polished)
                if len(radii) > 0:
                    max_r = np.max(radii)
                    if max_r < best_radius:
                        best_radius = max_r
                        best_points = polished.copy()
                    
        # Nelder-Mead style refinement
        if time.time() - start_time < time_limit - 10.0:
            nm_time = min(8.0, time_limit - (time.time() - start_time) - 5.0)
            if nm_time > 1.0:
                refined = nelder_mead_refine(points, nm_time)
                radii, _, _ = compute_facet_data(refined)
                if len(radii) > 0:
                    max_r = np.max(radii)
                    if max_r < best_radius:
                        best_radius = max_r
                        best_points = refined.copy()
    
    # Final Polish
    if best_points is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5.0:
            polished = polish(best_points, remaining_time)
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished.copy()
    else:
        best_points = init_fib
        
    return best_points
