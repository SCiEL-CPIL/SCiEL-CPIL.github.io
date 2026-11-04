import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

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
    
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[:] = [0, 0, 1]
    
    valid = norms > 1e-10
    
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
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def get_covering_radius(points):
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

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
        candidates = np.random.randn(1000, 3)
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

def smooth_covering_radius(points, temperature=0.01):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    """
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    
    # Log-sum-exp approximation to max
    max_r = np.max(radii)
    shifted = (radii - max_r) / temperature
    # Numerical stability: clip to prevent overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_r + temperature * log_sum
    return smooth_max

def nelder_mead_refine(points, indices, time_budget):
    """
    Refines a subset of points using Nelder-Mead to minimize the global covering radius.
    """
    start = time.time()
    initial_radius = get_covering_radius(points)
    
    # Flatten the subset of points
    x0 = points[indices].flatten()
    
    def objective(x_flat):
        if time.time() - start > time_budget:
            return initial_radius
        
        new_points = points.copy()
        new_points[indices] = x_flat.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= norms
        
        # Check for invalid states (coincident points etc)
        if np.any(norms < 1e-6):
            return np.pi
            
        # Use smooth objective for better gradient information
        return smooth_covering_radius(new_points, temperature=0.005)
        
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        # Verify improvement with exact radius
        new_points = points.copy()
        new_points[indices] = res.x.reshape(-1, 3)
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        exact_radius = get_covering_radius(new_points)
        
        if exact_radius < initial_radius:
            return new_points
    except Exception:
        pass
        
    return points

def weiszfeld_polish(points, time_budget, alpha_init=0.05):
    """
    Riemannian/Weiszfeld-style polish: iteratively move each point towards the
    area-weighted centroid of its adjacent facet circumcenters. This mimics the
    geometric median property of optimal covering configurations.
    """
    start = time.time()
    alpha = alpha_init
    best_local = points.copy()
    best_local_radius = get_covering_radius(points)
    
    for step in range(2000):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_radius = np.max(radii)
        if max_radius < best_local_radius:
            best_local_radius = max_radius
            best_local = points.copy()
            
        # Compute target for each point: weighted average of circumcenters of adjacent facets
        # Weight by 1/r^2 or similar to emphasize large facets
        targets = np.zeros_like(points)
        weights = np.zeros(N)
        
        # Inverse square weight to focus on large facets
        # Use a soft max to avoid division by zero for small facets
        r_min = np.min(radii)
        if r_min < 1e-6:
            r_min = 1e-6
            
        # We want points with large adjacent facets to move more towards the center
        # Actually, Weiszfeld for geometric median says: move to weighted centroid of neighbors
        # Here neighbors are facets.
        
        for m in range(len(radii)):
            w = 1.0 / (radii[m] ** 2)
            idx = simplices[m]
            targets[idx] += w * centers[m]
            weights[idx] += w
            
        valid = weights > 1e-10
        if not np.any(valid):
            break
            
        targets[valid] /= weights[valid, np.newaxis]
        
        # Move towards target
        # Use step size alpha
        points[valid] = points[valid] + alpha * (targets[valid] - points[valid])
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay alpha
        alpha *= 0.995
        
        if alpha < 1e-5:
            break
            
    return best_local

def polish(points, time_budget):
    """
    High-precision polish pass.
    Combines Weiszfeld-style centroid moves with Nelder-Mead refinement.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = get_covering_radius(points)
    
    # Phase 1: Weiszfeld-style polish
    if time.time() - start < time_budget * 0.6:
        t1 = time_budget * 0.6 - (time.time() - start)
        if t1 > 0.5:
            polished1 = weiszfeld_polish(points, t1, alpha_init=0.1)
            r1 = get_covering_radius(polished1)
            if r1 < best_local_radius:
                best_local = polished1
                best_local_radius = r1
                
    # Phase 2: Standard gradient-like descent (from original)
    if time.time() - start < time_budget * 0.85:
        t2 = time_budget * 0.85 - (time.time() - start)
        if t2 > 0.5:
            alpha = 0.005
            current = best_local.copy()
            for step in range(1000):
                if time.time() - start > time_budget * 0.85:
                    break
                    
                radii, centers, simplices = compute_facet_data(current)
                if len(radii) == 0:
                    break
                    
                max_radius = np.max(radii)
                if max_radius < best_local_radius:
                    best_local_radius = max_radius
                    best_local = current.copy()
                    
                n_worst = max(3, len(radii) // 10)
                worst_indices = np.argsort(radii)[-n_worst:]
                mask = np.zeros(len(radii), dtype=bool)
                mask[worst_indices] = True
                
                if not np.any(mask):
                    break
                    
                updates = np.zeros_like(current)
                
                for m in np.where(mask)[0]:
                    simplex = simplices[m]
                    i_idx, j_idx, k_idx = simplex
                    
                    a = current[i_idx]
                    b = current[j_idx]
                    c = current[k_idx]
                    
                    center = centers[m]
                    
                    step_size = alpha * radii[m]
                    
                    updates[i_idx] += step_size * (center - a)
                    updates[j_idx] += step_size * (center - b)
                    updates[k_idx] += step_size * (center - c)
                    
                current += updates
                current /= np.linalg.norm(current, axis=1, keepdims=True)
                
                alpha *= 0.998
                if alpha < 1e-6:
                    break
                    
    # Phase 3: Nelder-Mead refinement on worst points
    if time.time() - start < time_budget - 1.0:
        t3 = time_budget - 1.0 - (time.time() - start)
        if t3 > 1.0:
            radii, _, simplices = compute_facet_data(best_local)
            if len(radii) > 0:
                n_worst = max(5, len(radii) // 5)
                worst_facets = np.argsort(radii)[-n_worst:]
                worst_indices = set()
                for f in worst_facets:
                    worst_indices.update(simplices[f].tolist())
                worst_indices = list(worst_indices)
                
                if worst_indices:
                    refined = nelder_mead_refine(best_local, worst_indices, t3)
                    ref_radius = get_covering_radius(refined)
                    if ref_radius < best_local_radius:
                        best_local = refined
                        best_local_radius = ref_radius
                        
    return best_local

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
    
    # More randomized restarts with varied noise levels
    for s in range(20):
        np.random.seed(1000 + s * 7)
        noise_scale = 0.05 + 0.15 * (s / 20)
        init = init_fib + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = 25
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 137)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # Local Optimization
        alpha = 0.03
        no_improve = 0
        stuck_counter = 0
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
                stuck_counter = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    alpha *= 0.92
                    if alpha < 1e-5:
                        break
                if no_improve > 150:
                    # Try a small random perturbation to escape local minimum
                    np.random.seed(step * 31 + restart * 17)
                    perturbation = np.random.randn(N, 3) * 0.01
                    points = points + perturbation
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    no_improve = 0
                    stuck_counter += 1
                    if stuck_counter > 5:
                        break
                        
            # Identify facets to move - top 15% or at least 5 facets
            n_worst = max(5, len(radii) // 7)
            worst_indices = np.argsort(radii)[-n_worst:]
            mask = np.zeros(len(radii), dtype=bool)
            mask[worst_indices] = True
            
            if not np.any(mask):
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
                
            # Accumulate moves for vertices with weighted steps
            updates = np.zeros_like(points)
            
            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                center = centers[m]
                
                # Weight by how much this facet is above average
                avg_radius = np.mean(radii)
                weight = 1.0 + 0.5 * max(0, (radii[m] - avg_radius) / avg_radius)
                step_size = alpha * radii[m] * weight
                
                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)
                
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
    # Final polish pass on the best configuration found
    if best_points is not None and time.time() - start_time < time_limit - 3.0:
        remaining_time = time_limit - 3.0 - (time.time() - start_time)
        if remaining_time > 3.0:
            polished = polish(best_points, remaining_time)
            # Check if polished is better
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                pol_radius = np.max(radii)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished
                    
    if best_points is None:
        best_points = init_fib
        
    return best_points
