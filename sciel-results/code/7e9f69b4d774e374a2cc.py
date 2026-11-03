import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
CUBE_LIMIT = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(CUBE_LIMIT - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    min_wall_dist = np.min(CUBE_LIMIT - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _top_k_distance_objective(x, k, alpha):
    """
    Objective: Minimize the negative of a power-sum of the smallest distances.
    This effectively maximizes the minimum distance by focusing on the 'bottleneck' pairs.
    
    x: flattened centers (N*3,)
    k: number of smallest distances to consider
    alpha: exponent for the power sum (higher alpha focuses more on the smallest)
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    dists = np.sqrt(sq_dists)
    
    # Wall distances
    wall_dists = CUBE_LIMIT - np.abs(centers)
    
    # Combine pairwise and wall distances into a single list
    # We want to maximize the minimum of all these.
    # Using a power sum of the smallest k values.
    
    # Flatten pairwise distances (excluding diagonal)
    pair_dists = dists[np.triu_indices(N, k=1)]
    
    # Combine
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Get the smallest k distances
    # For large N, sorting is O(M log M). M ~ 5000. This is fast.
    # We can use np.partition to find the k smallest in O(M).
    if k < len(all_dists):
        smallest = np.partition(all_dists, k)[:k]
    else:
        smallest = all_dists
        
    # To avoid division by zero or issues with non-differentiability at 0,
    # we add a small epsilon.
    eps = 1e-6
    smallest = np.maximum(smallest, eps)
    
    # Power sum: sum(d^alpha). Minimize -sum(d^alpha).
    # Gradient calculation is tricky with partition, so we rely on finite differences
    # or a differentiable approximation.
    # However, L-BFGS-B requires gradients.
    # A common differentiable surrogate for min is the soft-min, but we want to focus on the top-k.
    # Let's use a weighted sum approach where weights are determined by the distances.
    
    # Alternative: Use the "log-sum-exp" trick for the minimum, but that considers all.
    # Let's stick to the soft-min but with a high beta, or use the top-k idea with a smooth weighting.
    
    # Actually, a very effective differentiable objective is:
    # f(x) = - ( sum_{i in TopK} d_i^p )^(1/p)
    # This is differentiable.
    
    p = alpha
    val = np.sum(smallest**p)
    obj = - (val ** (1.0/p))
    
    # Gradient
    # d(obj)/d(x) = - (1/p) (sum d^p)^(1/p - 1) * sum (p * d^(p-1) * d(d)/dx)
    # d(obj)/d(x) = - (sum d^p)^(1/p - 1) * sum (d^(p-1) * grad_d)
    
    # We need gradients of the distances.
    # This is computationally expensive to do exactly for the partitioned set.
    # Let's use a simpler, robust approach:
    # Minimize the sum of squared violations of a target radius r.
    # But we don't know r.
    
    # Let's go back to the soft-min but optimize the beta schedule carefully.
    # The previous seed used soft-min. Why did it score 1.0? Maybe it got stuck.
    # The "Top-K" idea is good but hard to differentiate.
    
    # Let's try a different differentiable objective:
    # Minimize the sum of exp(-beta * d_i) for all i.
    # This is the standard soft-min.
    # To focus on the "critical" pairs, we can use a high beta.
    
    # Let's implement a robust soft-min with gradient.
    
    # Pairwise
    finite_mask = np.isfinite(sq_dists)
    # Shift for numerical stability
    max_sq = np.max(sq_dists[finite_mask])
    shifted = sq_dists - max_sq
    exp_terms = np.exp(-beta * shifted)
    exp_terms[~finite_mask] = 0.0
    sum_exp = np.sum(exp_terms)
    
    if sum_exp > 0:
        log_sum = np.log(sum_exp)
        soft_min_sq = max_sq - (1.0 / beta) * log_sum
    else:
        soft_min_sq = max_sq
        
    # Wall
    wall_sq_dists = wall_dists**2
    max_wall_sq = np.max(wall_sq_dists)
    shifted_wall = wall_sq_dists - max_wall_sq
    exp_wall = np.exp(-beta * shifted_wall)
    sum_wall = np.sum(exp_wall)
    
    if sum_wall > 0:
        log_sum_wall = np.log(sum_wall)
        soft_min_wall_sq = max_wall_sq - (1.0 / beta) * log_sum_wall
    else:
        soft_min_wall_sq = max_wall_sq
        
    # Combine
    vals = np.array([soft_min_sq, soft_min_wall_sq])
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_vals = np.sum(exp_vals)
    
    if sum_vals > 0:
        log_sum_vals = np.log(sum_vals)
        total_soft_min_sq = max_val - (1.0 / beta) * log_sum_vals
    else:
        total_soft_min_sq = max_val
        
    total_soft_min_sq = max(total_soft_min_sq, 0.0)
    obj_val = -np.sqrt(total_soft_min_sq)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        # weights is (N, N)
        # grad_pair_sq[i] = sum_j weights[i,j] * (c_i - c_j)
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        # Wall gradient
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        d_wall_sq_dc = -2.0 * wall_dists * signs
        
        weights_wall = exp_wall / sum_wall
        grad_wall_sq = weights_wall[:, np.newaxis] * d_wall_sq_dc
        
        # Combine gradients
        w_A = np.exp(-beta * (soft_min_sq - max_val)) / sum_vals
        w_B = np.exp(-beta * (soft_min_wall_sq - max_val)) / sum_vals
        
        combined_grad_sq = w_A * grad_pair_sq + w_B * grad_wall_sq
        
        if total_soft_min_sq > 1e-12:
            grad = combined_grad_sq / (2.0 * np.sqrt(total_soft_min_sq))
        else:
            grad = np.zeros_like(centers)
            
    return obj_val, -grad # We want to MINIMIZE -radius, so return -obj_val? 
    # Wait. _compute_radius returns r. We want to MAXIMIZE r.
    # minimize(f) finds min f. So f should be -r.
    # obj_val is -sqrt(...). So return obj_val.
    # The gradient of -r is -grad_r.
    # My grad calculation above is for d(soft_min_sq)/dx.
    # r = sqrt(soft_min_sq).
    # dr/dx = (1/(2r)) * d(soft_min_sq)/dx.
    # d(-r)/dx = -(1/(2r)) * d(soft_min_sq)/dx.
    # My `grad` variable holds d(soft_min_sq)/dx (scaled).
    # So I should return -grad.
    
    return obj_val, -grad

def _optimize_packing(init_centers, time_limit=10.0):
    """
    Optimize packing from initial centers using soft-min L-BFGS-B.
    """
    start_time = time.time()
    x0 = init_centers.flatten()
    
    # Beta schedule: start small (smooth), end large (sharp)
    betas = [20, 50, 100, 200, 500, 1000, 2000]
    
    best_x = x0
    best_val = _compute_radius(init_centers)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _top_k_distance_objective(x, 10, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            x0 = result.x
            r = _compute_radius(x0.reshape(N, 3))
            if r > best_val:
                best_val = r
                best_x = x0
        except Exception:
            pass
            
    return best_x.reshape(N, 3)

def _refine_critical_pairs(centers, time_limit=5.0):
    """
    Refine by optimizing only the critical pairs (those close to the minimum distance).
    This is a local search on the contact graph.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        
        r = _compute_radius(centers)
        
        # Find critical pairs
        tree = cKDTree(centers)
        # Pairs with distance < 2*r * 1.1
        pairs = tree.query_pairs(2 * r * 1.1)
        
        # Wall contacts
        wall_contacts = []
        for i in range(N):
            for k in range(3):
                if CUBE_LIMIT - abs(centers[i][k]) < r * 1.1:
                    wall_contacts.append((i, k))
        
        if not pairs and not wall_contacts:
            break
            
        # Construct a smaller optimization problem
        # We want to maximize the minimum distance among these critical constraints.
        # Let's define a function that takes the positions of the spheres involved
        # and maximizes the min distance.
        
        # For simplicity, we can just do a few steps of gradient ascent on the full system
        # but with a very high beta in the soft-min, effectively focusing on the min.
        
        # Let's try a local move:
        # For each critical pair, try to move them apart slightly.
        improved = False
        for i, j in list(pairs)[:20]:
            if time.time() - start_time > time_limit:
                break
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10: continue
            direction = d / dist
            # Move i and j apart by a small amount
            step = 0.001
            new_centers = centers.copy()
            new_centers[i] += step * direction
            new_centers[j] -= step * direction
            new_centers = np.clip(new_centers, -CUBE_LIMIT, CUBE_LIMIT)
            
            new_r = _compute_radius(new_centers)
            if new_r > best_radius:
                centers = new_centers
                best_radius = new_r
                improved = True
                
        if not improved:
            break
            
    return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.01):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializations
    inits = []
    
    # 1. Grid
    inits.append(_grid_init(N))
    
    # 2. Jittered grids
    for seed in range(10):
        inits.append(_jitter_init(N, seed=seed, jitter=0.02))
        
    # 3. Random inits
    for seed in range(10):
        inits.append(_random_init(N, seed=seed))
        
    # Optimize each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            time_remaining = deadline - time.time() - 5.0
            if time_remaining < 5.0:
                break
                
            # Optimize with gradient descent
            optimized = _optimize_packing(init, time_limit=min(10.0, time_remaining * 0.6))
            r_opt = _compute_radius(optimized)
            
            # Refine critical pairs
            if time.time() < deadline - 10.0:
                refined = _refine_critical_pairs(optimized, time_limit=min(5.0, deadline - time.time() - 5.0))
                r_refined = _compute_radius(refined)
                if r_refined > r_opt:
                    optimized = refined
                    r_opt = r_refined
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
                
        except Exception:
            continue
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 3.0:
        try:
            polished = _refine_critical_pairs(best_centers, time_limit=min(5.0, deadline - time.time() - 2.0))
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
        except Exception:
            pass
            
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
