"""Rewritten program for sphere packing with improved optimization and polishing."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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

def _softmin_objective_and_grad(x, beta):
    """
    Objective: minimize negative soft-min of distances (pairwise and wall).
    x: flattened centers (N*3,)
    beta: inverse temperature (larger beta = sharper min)
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances squared
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances
    wall_dists = CUBE_LIMIT - np.abs(centers)
    wall_sq_dists = wall_dists**2
    
    # Soft-min on squared distances for pairs
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
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
    else:
        soft_min_sq = 0.0
        
    # Wall part
    max_wall_sq = np.max(wall_sq_dists)
    shifted_wall = wall_sq_dists - max_wall_sq
    exp_wall = np.exp(-beta * shifted_wall)
    sum_wall = np.sum(exp_wall)
    
    if sum_wall > 0:
        log_sum_wall = np.log(sum_wall)
        soft_min_wall_sq = max_wall_sq - (1.0 / beta) * log_sum_wall
    else:
        soft_min_wall_sq = max_wall_sq
        
    # Combine pair and wall soft-mins using another soft-min
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
    obj_val = np.sqrt(total_soft_min_sq)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Gradient of soft_min_sq_pair
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        # d(soft_min_sq_pair)/d(c_i) = sum_j weights[i,j] * (c_i - c_j)
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        # Wall gradient
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        d_wall_sq_dc = -2.0 * wall_dists * signs
        
        weights_wall = exp_wall / sum_wall
        grad_wall_sq = np.zeros_like(centers)
        for i in range(N):
            w = weights_wall[i]
            if w > 0:
                grad_wall_sq[i] += w * d_wall_sq_dc[i]
        grad_wall_sq *= -1.0 / beta
        
        # Combine gradients
        w_A = np.exp(-beta * (soft_min_sq - max_val)) / sum_vals
        w_B = np.exp(-beta * (soft_min_wall_sq - max_val)) / sum_vals
        
        combined_grad_sq = w_A * grad_pair_sq + w_B * grad_wall_sq
        
        if total_soft_min_sq > 1e-12:
            grad = combined_grad_sq / (2.0 * obj_val)
        else:
            grad = np.zeros_like(centers)
            
    # We want to MAXIMIZE radius, so MINIMIZE -radius.
    return -obj_val, -grad

def _optimize_packing(init_centers, time_limit=10.0):
    """
    Optimize packing from initial centers using soft-min L-BFGS-B.
    """
    start_time = time.time()
    x0 = init_centers.flatten()
    
    # Beta schedule: start small (smooth), end large (sharp)
    betas = [10, 50, 100, 500, 1000, 2000, 5000]
    
    best_x = x0
    best_val = _compute_radius(init_centers)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            # Check for NaN
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            x0 = result.x
            r = _compute_radius(x0.reshape(N, 3))
            if r > best_val:
                best_val = r
                best_x = x0
        except Exception:
            pass
            
    return best_x.reshape(N, 3)

def _lloyd_polish(centers, iterations=20, time_limit=5.0):
    """
    Lloyd's algorithm polish: move each center to the centroid of its Voronoi cell.
    This helps in balancing the distances.
    """
    start_time = time.time()
    centers = centers.copy()
    
    for _ in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        try:
            tri = Delaunay(centers)
            # For each point, find its neighbors in the Delaunay triangulation
            # and move it towards the centroid of those neighbors.
            new_centers = centers.copy()
            for i in range(N):
                neighbors = tri.simplices[tri.find_simplex(centers[i])]
                if len(neighbors) > 0:
                    # Centroid of neighbors
                    centroid = np.mean(centers[neighbors], axis=0)
                    # Move halfway towards centroid (damping)
                    new_centers[i] = centers[i] + 0.5 * (centroid - centers[i])
                else:
                    new_centers[i] = centers[i]
                    
            # Clip to cube
            new_centers = np.clip(new_centers, -CUBE_LIMIT, CUBE_LIMIT)
            
            # Accept if radius improved
            r_old = _compute_radius(centers)
            r_new = _compute_radius(new_centers)
            if r_new > r_old:
                centers = new_centers
            else:
                # Try smaller step
                new_centers = centers + 0.1 * (new_centers - centers)
                new_centers = np.clip(new_centers, -CUBE_LIMIT, CUBE_LIMIT)
                r_new = _compute_radius(new_centers)
                if r_new > r_old:
                    centers = new_centers
                else:
                    break
        except Exception:
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
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializations
    inits = []
    
    # 1. Grid
    inits.append(_grid_init(N))
    
    # 2. Jittered grids
    for seed in range(5):
        inits.append(_jitter_init(N, seed=seed, jitter=0.02))
        
    # 3. Random inits
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))
        
    # 4. Body-centered inspired
    rng = np.random.RandomState(42)
    bcc = np.zeros((N, 3))
    for i in range(N):
        bcc[i] = rng.uniform(-0.4, 0.4)
    inits.append(bcc)
    
    # Optimize each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            time_remaining = deadline - time.time() - 5.0
            if time_remaining < 5.0:
                break
                
            optimized = _optimize_packing(init, time_limit=min(20.0, time_remaining))
            r_opt = _compute_radius(optimized)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
                
        except Exception:
            continue
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            polished = _lloyd_polish(best_centers, iterations=50, time_limit=min(5.0, deadline - time.time() - 2.0))
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
