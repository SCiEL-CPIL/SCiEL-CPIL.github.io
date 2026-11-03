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
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, handle gracefully
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = np.random.randn(500, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = np.array(points)
        # Dot products
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)

def fixed_topology_optimize(points, max_steps=500, time_limit=None, start_time=None):
    """
    Optimize points using a fixed topology (convex hull structure) and 
    a smooth loss that penalizes variance in facet circumradii.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 140.0
    
    # Get initial hull structure
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return points, np.pi
    
    # Fixed topology optimization with variance reduction
    alpha = 0.02
    tau_var = 0.05  # Temperature for variance penalty
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        # Compute loss: max radius + variance penalty
        mean_r = np.mean(radii)
        var_r = np.var(radii)
        loss = np.max(radii) + tau_var * var_r
        
        # Compute gradient of the smooth surrogate
        # Use soft-max for max term, and direct gradient for variance
        r_scaled = radii / tau_var
        r_max = np.max(r_scaled)
        exp_terms = np.exp(r_scaled - r_max)
        sum_exp = np.sum(exp_terms)
        weights_max = exp_terms / sum_exp
        
        # Variance gradient: d(var)/d(r_i) = 2*(r_i - mean_r)/M
        weights_var = 2.0 * (radii - mean_r) / len(radii)
        
        # Combined weights
        combined_weights = weights_max + tau_var * weights_var
        
        # Ensure weights are non-negative for the max part
        # For variance, we allow negative weights (to reduce large radii, increase small ones)
        
        updates = np.zeros_like(points)
        
        sin_r = np.sin(radii)
        sin_r[sin_r < 1e-6] = 1e-6
        
        for m in range(len(radii)):
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            
            center = centers[m]
            w = combined_weights[m]
            factor = w / sin_r[m]
            
            updates[i_idx] += factor * center
            updates[j_idx] += factor * center
            updates[k_idx] += factor * center
        
        # Project to tangent plane
        for i in range(N):
            proj = np.dot(updates[i], points[i])
            updates[i] -= proj * points[i]
        
        points = points + alpha * updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            alpha *= 0.95
    
    return points, np.max(compute_facet_data(points)[0])

def nelder_mead_polish(points, time_limit=140.0, start_time=None):
    """
    Nelder-Mead style polish for the covering radius.
    """
    if start_time is None:
        start_time = time.time()
    
    def objective(p):
        p = p.reshape(N, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        radii, _, _ = compute_facet_data(p)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
    
    # Use scipy's minimize with Nelder-Mead
    from scipy.optimize import minimize
    
    x0 = points.flatten()
    
    def obj_flat(x):
        p = x.reshape(N, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        radii, _, _ = compute_facet_data(p)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
    
    try:
        result = minimize(obj_flat, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        best_p = result.x.reshape(N, 3)
        best_p = best_p / np.linalg.norm(best_p, axis=1, keepdims=True)
        return best_p
    except Exception:
        return points

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
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
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 3. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 4. Another greedy with different seed
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # More randomized restarts
    for s in range(15):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 15)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 137)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Phase 1: Fixed-topology optimization with variance reduction
        if time.time() - start_time < time_limit - 30.0:
            points, radius = fixed_topology_optimize(points, max_steps=800, 
                                                      time_limit=time_limit,
                                                      start_time=start_time)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
        
        # Phase 2: Nelder-Mead polish
        if time.time() - start_time < time_limit - 10.0:
            points_polished = nelder_mead_polish(points, time_limit=time_limit - 5.0,
                                                 start_time=start_time)
            radii_final, _, _ = compute_facet_data(points_polished)
            if len(radii_final) > 0:
                radius_final = np.max(radii_final)
                if radius_final < best_radius:
                    best_radius = radius_final
                    best_points = points_polished.copy()
        
        # Phase 3: Quick Lloyd-style refinement on best points
        if time.time() - start_time < time_limit - 5.0:
            points_lloyd = best_points.copy()
            alpha_lloyd = 0.01
            for step in range(200):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                radii, centers, simplices = compute_facet_data(points_lloyd)
                if len(radii) == 0:
                    break
                
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points_lloyd.copy()
                
                # Move only the top 15% largest facets
                if len(radii) > 5:
                    threshold = np.percentile(radii, 85)
                    mask = radii > threshold
                else:
                    max_idx = np.argmax(radii)
                    mask = np.zeros(len(radii), dtype=bool)
                    mask[max_idx] = True
                    
                if not np.any(mask):
                    break
                    
                updates = np.zeros_like(points_lloyd)
                
                for m in np.where(mask)[0]:
                    simplex = simplices[m]
                    i_idx, j_idx, k_idx = simplex
                    
                    center = centers[m]
                    step_size = alpha_lloyd * radii[m]
                    
                    updates[i_idx] += step_size * (center - points_lloyd[i_idx])
                    updates[j_idx] += step_size * (center - points_lloyd[j_idx])
                    updates[k_idx] += step_size * (center - points_lloyd[k_idx])
                    
                # Project to tangent
                for i in range(N):
                    proj = np.dot(updates[i], points_lloyd[i])
                    updates[i] -= proj * points_lloyd[i]
                
                points_lloyd += updates
                points_lloyd /= np.linalg.norm(points_lloyd, axis=1, keepdims=True)
                
                alpha_lloyd *= 0.995

    if best_points is None:
        best_points = init_fib
        
    return best_points
