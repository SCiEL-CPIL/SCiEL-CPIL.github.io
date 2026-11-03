import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """Compute circumradius and circumcenter for each facet."""
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
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
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

def greedy_init(n_points, seed=42):
    """Farthest Point Sampling initialization."""
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(5000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
        
    return np.array(points)

def icosahedral_init(n_points, seed=42):
    """Generate points with icosahedral symmetry as starting structure."""
    np.random.seed(seed)
    
    # Start with icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add more points by subdividing faces
    points = list(ico)
    
    while len(points) < n_points:
        # Find the largest facet and add a point near its center
        pts = np.array(points)
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            break
        max_idx = np.argmax(radii)
        center = centers[max_idx]
        # Add point slightly perturbed from center
        new_pt = center + np.random.randn(3) * 0.01
        new_pt /= np.linalg.norm(new_pt)
        points.append(new_pt)
    
    return np.array(points[:n_points])

def soft_max_objective(points, beta=50.0):
    """
    Compute smooth surrogate for max facet radius using log-sum-exp.
    Returns value and gradient.
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Use squared radii for the soft max
    r2 = radii ** 2
    max_r2 = np.max(r2)
    scaled = beta * (r2 - max_r2)
    
    # Stable log-sum-exp
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    log_sum_exp = beta * max_r2 + max_scaled + np.log(sum_exp)
    value = np.sqrt(log_sum_exp / beta)
    
    # Gradient computation
    grad = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-10:
            continue
        
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-8:
            continue
        
        # d(r_i)/d(p_j) for j in {i, j, k} of facet m
        # r_i = arccos(u_i . p_i)
        # dr_i = -(u_i . dp_i) / sin(r_i)
        coeff = -weights[m] * 2.0 * beta * r / sin_r
        
        grad[i_idx] += coeff * center
        grad[j_idx] += coeff * center
        grad[k_idx] += coeff * center
    
    if value > 1e-12:
        grad /= (2.0 * value * beta)
    
    return value, grad

def polish_gradient(points, time_budget, start_alpha=0.1):
    """High-precision gradient polish using smooth surrogate."""
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    alpha = start_alpha
    
    for step in range(5000):
        if time.time() - start > time_budget:
            break
        
        val, grad = soft_max_objective(points, beta=50.0)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Take step
        points = points + alpha * grad_tangent
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check actual covering radius
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
        
        # Decay step size
        alpha *= 0.995
        if alpha < 1e-6:
            break
    
    return best_local

def polish_facet_push(points, time_budget):
    """Final polish by pushing vertices of largest facets."""
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    T = 0.05
    
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
        
        # Find facets above threshold
        threshold = np.percentile(radii, 85)
        mask = radii > threshold
        
        if not np.any(mask):
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
        
        # Push vertices toward circumcenters
        updates = np.zeros_like(points)
        for m in np.where(mask)[0]:
            idx = simplices[m]
            i_idx, j_idx, k_idx = idx
            center = centers[m]
            r = radii[m]
            
            step_size = T * r
            updates[i_idx] += step_size * (center - points[i_idx])
            updates[j_idx] += step_size * (center - points[j_idx])
            updates[k_idx] += step_size * (center - points[k_idx])
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        T *= 0.99
        if T < 1e-6:
            break
    
    return best_local

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(("fib", init_fib))
    
    # 2. Greedy (FPS)
    init_greedy = greedy_init(N, seed=10)
    initializations.append(("greedy10", init_greedy))
    
    # 3. Greedy with different seed
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(("greedy123", init_greedy2))
    
    # 4. Icosahedral-based
    init_ico = icosahedral_init(N, seed=42)
    initializations.append(("ico42", init_ico))
    
    # 5. Icosahedral with different seed
    init_ico2 = icosahedral_init(N, seed=77)
    initializations.append(("ico77", init_ico2))
    
    # 6. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.15
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(("fib_noise", init_fib_noise))
    
    # 7. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(("rand42", init_rand))
    
    # 8. Greedy with more candidates
    np.random.seed(999)
    candidates = np.random.randn(20000, 3)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    # Simple greedy selection
    pts = [candidates[0]]
    for _ in range(N - 1):
        existing = np.array(pts)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        best_idx = np.argmax(max_dist)
        pts.append(candidates[best_idx])
    init_greedy3 = np.array(pts)
    initializations.append(("greedy999", init_greedy3))
    
    # Optimization for each initialization
    for init_name, init_points in initializations:
        if time.time() - start_time > time_limit - 15.0:
            break
        
        points = init_points.copy()
        
        # Phase 1: Facet push optimization (main workhorse)
        T = 0.08
        no_improve = 0
        local_best = points.copy()
        local_best_r = np.pi
        
        for step in range(2000):
            if time.time() - start_time > time_limit - 15.0:
                break
            
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break
            
            max_r = np.max(radii)
            if max_r < local_best_r:
                local_best_r = max_r
                local_best = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    break
            
            # Push vertices of large facets
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
                
                step_size = T * r
                updates[i_idx] += step_size * (center - points[i_idx])
                updates[j_idx] += step_size * (center - points[j_idx])
                updates[k_idx] += step_size * (center - points[k_idx])
            
            points = points + updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            T *= 0.995
        
        # Phase 2: Gradient polish
        if time.time() - start_time < time_limit - 10.0:
            polished = polish_gradient(local_best, time_budget=min(8.0, time_limit - 10.0 - (time.time() - start_time)))
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < local_best_r:
                    local_best_r = max_r
                    local_best = polished
        
        # Phase 3: Facet push polish
        if time.time() - start_time < time_limit - 5.0:
            polished = polish_facet_push(local_best, time_budget=min(5.0, time_limit - 5.0 - (time.time() - start_time)))
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < local_best_r:
                    local_best_r = max_r
                    local_best = polished
        
        # Update global best
        if local_best_r < best_radius:
            best_radius = local_best_r
            best_points = local_best.copy()
        
        # Print progress for debugging
        print(f"Init {init_name}: radius = {np.degrees(local_best_r):.6f} deg")
    
    # Final polish on the best result
    if best_points is not None and time.time() - start_time < time_limit - 3.0:
        remaining = time_limit - 3.0 - (time.time() - start_time)
        if remaining > 2.0:
            polished = polish_gradient(best_points, time_budget=remaining)
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    
    if best_points is None:
        best_points = initializations[0][1]
    
    # Final verification
    radii, _, _ = compute_facet_data(best_points)
    if len(radii) > 0:
        final_r = np.max(radii)
        print(f"Final radius: {np.degrees(final_r):.6f} deg")
    
    return best_points
