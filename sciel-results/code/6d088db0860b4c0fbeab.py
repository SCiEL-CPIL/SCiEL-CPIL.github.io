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
        # For valid facets, compute normally, for invalid keep defaults
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
        # Generate candidates
        candidates = np.random.randn(1000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Find min distance to existing points
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
        # If no facets are above threshold, pick the single largest
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
    Returns:
        value: float
        grad: array of shape (N, 3)
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Scale radii to make log-sum-exp stable
    # We want to approximate max(radii)
    # Let f_i = radii_i^2. Then max(radii) = sqrt(max(f_i)).
    # logsumexp(f) approximates log(sum(exp(f))).
    # If we use beta * f_i, we get a soft max.
    
    beta = 100.0 # Sharpness of the soft max
    
    # To avoid overflow, subtract max
    max_r2 = np.max(radii**2)
    scaled = beta * (radii**2 - max_r2)
    
    # logsumexp
    # LSE = max_r2*beta + log(sum(exp(scaled)))
    # Value = sqrt(max_r2 + LSE/beta) ? No.
    # Let S = sum(exp(beta * r_i^2)).
    # log(S) = beta * max_r2 + log(sum(exp(beta*(r_i^2 - max_r2))))
    # SoftMaxVal = sqrt( log(S) / beta )
    
    # Gradient of SoftMaxVal w.r.t points:
    # d/dp sqrt(log(S)/beta) = (1/(2*sqrt(log(S)/beta))) * (1/beta) * (1/S) * dS/dp
    # dS/dp = S * (1/S * dS/dp) = sum(exp(beta*r_i^2) * 2*beta*r_i * dr_i/dp)
    # So dS/dp / S = sum( w_i * 2*beta*r_i * dr_i/dp ) where w_i = exp(beta*r_i^2)/S
    
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # log_sum_exp_val = beta * max_r2 + max_scaled + log(sum_exp)
    # Actually, log(S) = beta * max_r2 + log(sum(exp(beta*r_i^2 - beta*max_r2)))
    # = beta * max_r2 + log(sum(exp(scaled)))
    
    log_sum_exp_val = beta * max_r2 + max_scaled + np.log(sum_exp)
    
    value = np.sqrt(log_sum_exp_val / beta)
    
    # Gradient
    # grad_val = (1 / (2 * value * beta)) * (1/S) * dS/dp
    # dS/dp = sum_i exp(beta*r_i^2) * 2*beta*r_i * dr_i/dp
    # (1/S) * dS/dp = sum_i w_i * 2*beta*r_i * dr_i/dp
    
    grad_points = np.zeros_like(points)
    
    # dr_i/dp for each facet i
    # r_i = arccos(u_i . a_i)
    # dr_i = - (u_i . dp_a) / sqrt(1 - (u_i . a_i)^2) = - (u_i . dp_a) / sin(r_i)
    # Note: u_i is fixed if we treat it as constant for this step (it depends on p, but this is a first-order approximation or we ignore the derivative of u).
    # Actually u_i depends on p. But for a soft-max gradient step, ignoring the derivative of the circumcenter direction is often a good enough approximation for a local step, or we can compute it.
    # Let's assume u is constant for the gradient calculation of the radius term, which is a common approximation in these geometric optimizers.
    
    # dr_i = - (u_i . (dp_i + dp_j + dp_k - 3*centroid? No, u is normal.))
    # Wait, r_i is distance from u_i to a_i.
    # If u_i is fixed, dr_i = - (u_i . d a_i) / sin(r_i).
    # The update to point p_a affects r_i for all facets containing a.
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        # Avoid division by zero if r is 0 or pi
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # Coefficient for d a
        # d r = - (u . d a) / sin(r)
        # Term in gradient sum: w_m * 2 * beta * r * ( - u / sin(r) )
        coeff = -weights[m] * 2.0 * beta * r / sin_r
        
        grad_points[i_idx] += coeff * center
        grad_points[j_idx] += coeff * center
        grad_points[k_idx] += coeff * center
        
    # Final scaling
    if value > 1e-12:
        grad_points /= (2.0 * value * beta)
    else:
        grad_points = np.zeros_like(points)
        
    return value, grad_points

def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass.
    """
    start = time.time()
    alpha = start_alpha
    best_local = points.copy()
    best_local_radius = np.pi
    
    # Phase 1: Gradient Descent on Smooth Surrogate
    for step in range(1000):
        if time.time() - start > time_budget * 0.5:
            break
            
        val, grad = smooth_objective_and_grad(points)
        
        # Compute actual max radius for tracking
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
                
        # Normalize gradient to unit sphere tangent space?
        # The gradient computed is in R3. We should project it to the tangent plane.
        # grad_tangent = grad - (grad . p) p
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Step
        points = points + alpha * grad_tangent
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha *= 0.99
        
        if alpha < 1e-5:
            break
            
    points = best_local
    
    # Phase 2: Facet Push with decaying step
    alpha_push = 0.02
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
            
        threshold = np.percentile(radii, 90)
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
        
        alpha_push *= 0.995
        
        if alpha_push < 1e-5:
            break
            
    return best_local

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
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
    
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Perturb the best found so far
            if best_points is not None:
                np.random.seed(restart * 137)
                points = best_points + np.random.randn(N, 3) * 0.05
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # Optimization loop
        T = 0.05 # Step size for facet push
        alpha_lloyd = 0.5 # Step size for Lloyd
        no_improve_count = 0
        current_radius = np.pi
        
        # Inner loop
        while True:
            if time.time() - start_time > time_limit:
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
            
            # Alternate steps
            if no_improve_count % 10 == 0:
                # Lloyd Step
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                # Facet Push Step
                points = facet_push_step(points, radii, centers, simplices, T=T, percentile=80)
                
            # Normalize just in case
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            # Cool down
            T *= 0.995
            alpha_lloyd *= 0.995
            
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
                    best_points = polished
    else:
        best_points = init_fib
        
    return best_points
