import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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

def get_objective_and_gradient(points, tau=0.1):
    """
    Computes the log-sum-exp objective and its gradient (descent direction).
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
        
    # Log-sum-exp for numerical stability
    r_scaled = radii / tau
    r_max = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    objective = np.log(sum_exp) + r_max
    
    # Gradient computation - descent direction
    updates = np.zeros_like(points)
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-6] = 1e-6
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        w = weights[m]
        factor = w / sin_r[m]
        
        updates[i_idx] += factor * center
        updates[j_idx] += factor * center
        updates[k_idx] += factor * center
        
    # Project updates to tangent plane (vectorized)
    proj = np.sum(updates * points, axis=1, keepdims=True)
    updates -= proj * points
        
    return objective, updates

def force_init(n_points, iterations=200, seed=42):
    """
    Initialize points using a repulsive force model to spread them out.
    Vectorized for speed.
    """
    np.random.seed(seed)
    points = np.random.randn(n_points, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    for it in range(iterations):
        # Compute pairwise distances
        dists_sq = 1.0 - points @ points.T
        np.fill_diagonal(dists_sq, np.inf)
        
        # Force: sum over j of (p_i - p_j) / |p_i - p_j|^3
        # |p_i - p_j|^2 = 2(1 - cos) = dists_sq
        # |p_i - p_j| = sqrt(dists_sq)
        # So force contribution = (p_i - p_j) / dists_sq^(3/2)
        
        # Vectorized: for each i, sum over j of (points[i] - points[j]) / dists_sq[i,j]^(3/2)
        # This is O(N^2) but vectorized
        # dists_sq has inf on diagonal, so 1/dists_sq^(3/2) will be 0 there
        inv_d3 = np.where(dists_sq > 1e-10, dists_sq ** (-1.5), 0.0)
        
        # forces[i] = sum_j (points[i] - points[j]) * inv_d3[i,j]
        # = points[i] * sum_j inv_d3[i,j] - sum_j points[j] * inv_d3[i,j]
        
        sum_inv = np.sum(inv_d3, axis=1, keepdims=True)
        weighted_sum = inv_d3 @ points  # shape (N, 3)
        
        forces = points * sum_inv - weighted_sum
        
        # Adaptive step size
        step = 0.05 * (1.0 / (1.0 + it * 0.01))
        
        points += step * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
    return points

def polish_worst_facets(points, max_time=5.0):
    """
    Local polish: focus on the worst facets using small moves toward circumcenters.
    """
    start = time.time()
    alpha = 0.01
    
    for step in range(200):
        if time.time() - start > max_time:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_radius = np.max(radii)
        
        # Move only the worst facets (top 10%)
        if len(radii) > 10:
            threshold = np.percentile(radii, 90)
            mask = radii > threshold
        else:
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
            
        if not np.any(mask):
            break
            
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            
            center = centers[m]
            step_size = alpha * radii[m]
            
            updates[i_idx] += step_size * (center - points[i_idx])
            updates[j_idx] += step_size * (center - points[j_idx])
            updates[k_idx] += step_size * (center - points[k_idx])
            
        # Project to tangent
        proj = np.sum(updates * points, axis=1, keepdims=True)
        updates -= proj * points
        
        points += updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha *= 0.98
        
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
    
    # 2. Force-based initialization (faster now)
    init_force = force_init(N, iterations=100, seed=42)
    initializations.append(init_force)
    
    # 3. Force-based with different seed
    init_force2 = force_init(N, iterations=100, seed=123)
    initializations.append(init_force2)
    
    # 4. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.15
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 5. More randomized restarts with varying noise
    for s in range(12):
        np.random.seed(1000 + s * 7)
        noise_scale = 0.05 + 0.15 * s / 12
        init = init_fib + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        points = initializations[restart].copy()
        
        # Phase 1: Gradient Descent with Line Search
        tau = 0.1
        alpha = 0.1
        no_improve = 0
        
        for step in range(3000):
            if time.time() - start_time > time_limit - 10.0:
                break
                
            obj, grad = get_objective_and_gradient(points, tau)
            
            # Check actual covering radius for best tracking
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) > 0:
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 150:
                        break
            else:
                break
            
            direction = grad  # Descent direction
            
            # Normalize direction
            norm_dir = np.linalg.norm(direction, axis=1)
            norm_dir[norm_dir < 1e-10] = 1e-10
            direction = direction / norm_dir[:, np.newaxis]
            
            # Backtracking line search
            step_size = alpha
            improved = False
            for _ in range(8):
                new_points = points + step_size * direction
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_obj, _ = get_objective_and_gradient(new_points, tau)
                
                if new_obj < obj:
                    points = new_points
                    improved = True
                    break
                else:
                    step_size *= 0.5
                    
            if not improved:
                alpha *= 0.5
                if alpha < 1e-4:
                    break
                    
        # Phase 2: Polish
        if time.time() - start_time < time_limit - 5.0:
            points = polish_worst_facets(points, max_time=3.0)
            
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) > 0:
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
