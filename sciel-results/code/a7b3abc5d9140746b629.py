import numpy as np
import time

N = 50

def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _soft_min_radius(centers, alpha=20.0):
    """
    Compute a smooth approximation of the minimum radius.
    r_soft = (sum of exp(-alpha * d_i)^(-1/alpha))^(1/alpha) / 2
    where d_i are all constraints (pair distances and wall distances).
    Actually, we want to maximize the minimum. 
    Let's use: r = min(d_ij/2, wall_i).
    Soft min of a set of values x_i is: (sum x_i^(-p))^(-1/p) for large p.
    Here x_i are the distances (d_ij/2 and wall_i).
    """
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # We only need upper triangle
    i, j = np.triu_indices(N, k=1)
    pair_dists = dist[i, j] / 2.0
    
    # Wall
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all constraints
    # We want to maximize min(constraints).
    # Soft min: (sum c_i^(-p))^(-1/p)
    # This is numerically unstable for large p.
    # Alternative: -logsumexp(-alpha * c) / alpha
    # Let's use that.
    all_constraints = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # Clip to avoid log(0) or negative values causing issues
    # If any constraint is <= 0, the score is 0 (or very negative).
    if np.any(all_constraints <= 1e-12):
        return 1e-12
    
    log_sum = np.log(np.sum(np.exp(-alpha * all_constraints)))
    return -log_sum / alpha

def _negative_soft_min(centers, alpha=20.0):
    """Objective to minimize: -soft_min_radius."""
    return -_soft_min_radius(centers, alpha)

def _soft_min_gradient(centers, alpha=20.0):
    """
    Gradient of -soft_min_radius with respect to centers.
    f(c) = - (1/alpha) * log( sum exp(-alpha * c_i) )
    df/dc_j = - (1/alpha) * (1/sum) * d/dc_j (sum exp(-alpha * c_i))
    df/dc_j = - (1/alpha) * (1/S) * sum_i ( -alpha * exp(-alpha * c_i) * dc_i/dc_j )
    df/dc_j = (1/S) * sum_i exp(-alpha * c_i) * dc_i/dc_j
    
    where S = sum exp(-alpha * c_i).
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    i, j = np.triu_indices(N, k=1)
    pair_dists = dist[i, j] / 2.0
    
    wall_dists = 0.5 - np.abs(centers)
    
    all_constraints = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # Clip
    all_constraints = np.maximum(all_constraints, 1e-12)
    
    exp_vals = np.exp(-alpha * all_constraints)
    S = np.sum(exp_vals)
    
    weights = exp_vals / S
    
    # Gradient of c_i with respect to centers
    # c_i for pairs: d_ij / 2. d_ij = ||c_i - c_j||.
    # d(d_ij)/dc_k = (c_k - c_other) / d_ij
    # d(c_i)/dc_k = 0.5 * (c_k - c_other) / d_ij
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    # weights[:N*(N-1)/2] correspond to pairs
    # We need to map back to indices
    # This is tricky with vectorization. Let's do it in a loop or careful index math.
    # For N=50, 1225 pairs. A loop is fine.
    
    # Actually, let's compute the gradient of the potential energy style.
    # Force on i due to j: w_ij * (c_i - c_j) / d_ij
    # Here w_ij is the weight of the constraint d_ij/2.
    # d(c_ij/2)/dc_i = (c_i - c_j) / (2 * d_ij)
    # d(c_ij/2)/dc_j = (c_j - c_i) / (2 * d_ij)
    
    pair_weights = weights[:len(i)]
    
    # Vectorized update for pairs
    # diff[i,j] = c_i - c_j
    # We want to add w * (c_i - c_j) / (2 * d_ij) to grad[i]
    # and w * (c_j - c_i) / (2 * d_ij) to grad[j]
    
    # dist[i,j] is d_ij
    # factor = pair_weights / (2 * dist[i,j])
    # Note: dist is the full matrix, but we extracted upper triangle.
    # Let's recompute dist for the pairs to be safe or use the matrix.
    # We have dist matrix.
    
    # Create a matrix of forces
    # force_matrix[i,j] = w_ij * (c_i - c_j) / (2 * d_ij)
    # This is only valid for i != j.
    
    # Let's just loop for simplicity and safety, N=50 is small.
    for k in range(len(i)):
        idx_i = i[k]
        idx_j = j[k]
        w = pair_weights[k]
        d = dist[idx_i, idx_j]
        if d < 1e-12:
            continue
        vec = (centers[idx_i] - centers[idx_j]) / (2.0 * d)
        grad[idx_i] += w * vec
        grad[idx_j] -= w * vec
        
    # Wall contributions
    # c_wall_k = 0.5 - |c_k|
    # d(c_wall_k)/dc_k = -sign(c_k)
    # So grad contribution is w_wall * (-sign(c_k))
    
    wall_weights = weights[len(i):]
    # wall_dists is shape (N, 3)
    # wall_weights is shape (N*3,)
    wall_weights_matrix = wall_weights.reshape(N, 3)
    
    # sign of centers
    signs = np.sign(centers)
    # where center is 0, sign is 0, but derivative is undefined. 
    # We can use 0 or subgradient. 0 is fine.
    
    grad -= wall_weights_matrix * signs
    
    return grad

def _optimize_smooth(centers, rng, steps=5000, alpha=20.0, lr=0.01):
    """
    Gradient descent on -soft_min_radius.
    """
    current = centers.copy()
    current_obj = _negative_soft_min(current, alpha)
    
    for step in range(steps):
        grad = _soft_min_gradient(current, alpha)
        
        # Adaptive learning rate?
        # Simple decay
        lr_eff = lr * (1.0 - step / steps) ** 0.5
        if lr_eff < 1e-5:
            lr_eff = 1e-5
            
        proposed = current - lr_eff * grad
        
        # Clip to bounds
        proposed = np.clip(proposed, -0.5, 0.5)
        
        proposed_obj = _negative_soft_min(proposed, alpha)
        
        if proposed_obj < current_obj:
            current = proposed
            current_obj = proposed_obj
        else:
            # If it gets worse, maybe reduce lr or skip
            # For now, just skip
            pass
            
    return current

def _local_search(centers, rng, steps=2000, t_start=0.005, t_end=0.0001):
    """
    Fine-tuned Simulated Annealing using the exact score (min radius).
    """
    current = centers.copy()
    current_score = _compute_score(current)
    
    cool_factor = (t_end / t_start) ** (1.0 / steps)
    
    for step in range(steps):
        T = t_start * (cool_factor ** step)
        
        # Pick a random point and move it
        idx = rng.integers(0, N)
        original = current[idx].copy()
        
        # Gaussian move
        noise_scale = 0.005
        move = rng.normal(0, noise_scale, size=3)
        proposed = current.copy()
        proposed[idx] = original + move
        proposed[idx] = np.clip(proposed[idx], -0.5, 0.5)
        
        proposed_score = _compute_score(proposed)
        
        delta = proposed_score - current_score
        
        if delta > 0:
            current = proposed
            current_score = proposed_score
        else:
            if T > 0:
                # Probability of accepting worse move
                # Scale delta for probability
                # delta is in radius units.
                prob = np.exp(-delta / (T * 0.01)) # T is small, so divide by small number? 
                # Actually, SA usually uses exp(-delta_E / T).
                # Here delta is improvement (positive is good).
                # So delta_E = -delta.
                # prob = exp(delta / T).
                # If delta is negative (worse), exp(negative/T) is small.
                # T is ~ 0.001 to 0.005.
                # delta is ~ 0.001.
                # exp(-0.001 / 0.001) = exp(-1) = 0.36.
                # This seems reasonable.
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current = proposed
                    current_score = proposed_score
                    
    return current

def _bcc_initial():
    """
    Initialize with a BCC-like lattice.
    """
    # 4x4x4 grid is 64 points.
    # BCC adds points at the center of the cubes.
    # Let's just use a jittered grid for now, it's simpler and works well.
    m = 4
    spacing = 1.0 / m
    grid_points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                grid_points.append([
                    -0.5 + (i + 0.5) * spacing,
                    -0.5 + (j + 0.5) * spacing,
                    -0.5 + (k + 0.5) * spacing
                ])
    return np.array(grid_points)

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin
    
    best_score = 0.0
    best_centers = None
    
    # Initialization
    grid_points = _bcc_initial()
    
    # Strategy: 
    # 1. Multiple restarts with different initializations (grid jitter, random)
    # 2. For each restart:
    #    a. Smooth optimization (gradient descent on soft-min)
    #    b. Local search (SA on exact score)
    
    num_restarts = 15
    rng_master = np.random.default_rng(123)
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        rng = np.random.default_rng(1000 + i)
        
        if i % 3 == 0:
            # Jittered grid
            indices = rng.permutation(len(grid_points))
            centers = grid_points[indices[:N]].copy()
            centers += rng.normal(0, 0.02, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
        elif i % 3 == 1:
            # Random
            centers = rng.uniform(-0.45, 0.45, (N, 3))
        else:
            # BCC specific: take 4x4x4 grid, add center points, select 50
            # This is complex to set up perfectly. Let's stick to grid/random.
            # Maybe a denser grid? 5x5x5 = 125 points.
            m = 5
            spacing = 1.0 / m
            gp = []
            for a in range(m):
                for b in range(m):
                    for c in range(m):
                        gp.append([-0.5 + (a + 0.5) * spacing,
                                   -0.5 + (b + 0.5) * spacing,
                                   -0.5 + (c + 0.5) * spacing])
            gp = np.array(gp)
            indices = rng.permutation(len(gp))
            centers = gp[indices[:N]].copy()
            centers += rng.normal(0, 0.01, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            
        # Phase 1: Smooth optimization
        # Use a moderate number of steps.
        # The gradient calculation is O(N^2) per step.
        # 5000 steps * 2500 ops = 12.5M ops. Fast.
        smooth_centers = _optimize_smooth(centers, rng, steps=3000, alpha=20.0, lr=0.05)
        
        # Phase 2: Local Search
        # Use remaining time?
        # Let's give it a fixed amount of steps, but check time.
        ls_steps = 2000
        if time.time() - start_time > time_limit - 20:
            ls_steps = 500
            
        ls_centers = _local_search(smooth_centers, rng, steps=ls_steps, t_start=0.002, t_end=0.00005)
        
        score = _compute_score(ls_centers)
        
        if score > best_score:
            best_score = score
            best_centers = ls_centers.copy()
            
    # Final Polish on the best candidate
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        rng_polish = np.random.default_rng(9999)
        # Do another round of smooth optimization with higher alpha (sharper)
        polished_smooth = _optimize_smooth(best_centers, rng_polish, steps=2000, alpha=50.0, lr=0.01)
        
        # Then local search
        polished_ls = _local_search(polished_smooth, rng_polish, steps=3000, t_start=0.001, t_end=0.00001)
        
        polish_score = _compute_score(polished_ls)
        
        if polish_score > best_score:
            best_score = polish_score
            best_centers = polished_ls
            
    if best_centers is None:
        # Fallback
        centers = np.zeros((N, 3))
        grid = np.array(np.meshgrid(np.linspace(-0.4, 0.4, 4), 
                                    np.linspace(-0.4, 0.4, 4), 
                                    np.linspace(-0.4, 0.4, 4))).T.reshape(-1, 3)
        centers = grid[:50]
        return centers

    return best_centers
