"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_radius_and_grad(x, beta=50.0):
    """
    Compute a smooth approximation to the minimum radius and its gradient.
    Uses log-sum-exp for soft-min of pairwise distances/2 and wall distances.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Only consider off-diagonal pairs
    mask = ~np.eye(N, dtype=bool)
    sq_dists_masked = np.where(mask, sq_dists, np.inf)
    
    # Wall distances
    walls = 0.5 - np.abs(centers)
    
    # All "radius candidates": pairwise dist/2 and wall distances
    # We want soft-min of these
    # For efficiency, compute separately then combine
    
    # Soft-min of pairwise distances / 2
    # log-sum-exp trick: soft_min(a) = -log(sum(exp(-beta*a))) / beta
    # But we need to be careful with large values
    
    # Pairwise: we have N*(N-1)/2 unique pairs, but let's use the full matrix
    # and take only upper triangle or all off-diagonal
    
    # For numerical stability, subtract max before exp
    sq_dists_finite = sq_dists_masked[np.isfinite(sq_dists_masked)]
    if len(sq_dists_finite) > 0:
        min_sq = np.min(sq_dists_finite)
        # Work with distances, not squared distances
        dists_offdiag = np.sqrt(sq_dists_masked)
        dists_offdiag = dists_offdiag / 2.0  # radius candidates from pairs
        
        # Soft-min using log-sum-exp
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # For stability: let m = min(d), then soft_min = m - 1/beta * log(sum(exp(-beta*(d-m))))
        d_min = np.min(dists_offdiag[np.isfinite(dists_offdiag)])
        exp_terms = np.exp(-beta * (dists_offdiag - d_min))
        exp_terms[~np.isfinite(exp_terms)] = 0.0
        log_sum = np.log(np.sum(exp_terms))
        soft_min_pair = d_min - log_sum / beta
    else:
        soft_min_pair = 0.5
    
    # Wall distances
    w_min = np.min(walls)
    exp_terms_wall = np.exp(-beta * (walls - w_min))
    exp_terms_wall[~np.isfinite(exp_terms_wall)] = 0.0
    log_sum_wall = np.log(np.sum(exp_terms_wall))
    soft_min_wall = w_min - log_sum_wall / beta
    
    # Combine: soft-min of (soft_min_pair, soft_min_wall)
    # This is approximate but works well
    all_candidates = np.array([soft_min_pair, soft_min_wall])
    c_min = np.min(all_candidates)
    exp_terms_comb = np.exp(-beta * (all_candidates - c_min))
    log_sum_comb = np.log(np.sum(exp_terms_comb))
    soft_min_radius = c_min - log_sum_comb / beta
    
    # Gradient computation
    # d(soft_min)/d(centers) requires chain rule through the log-sum-exp
    
    # For pairwise part:
    # soft_min_pair = d_min - (1/beta) * log(sum_j exp(-beta * d_j))
    # where d_j = ||c_i - c_j||/2 for all pairs
    # d(soft_min_pair)/d(c_i) = -sum_j [exp(-beta*d_j) / sum_k exp(-beta*d_k)] * d(d_j)/d(c_i)
    # d(d_j)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||) for pair (i,j)
    
    # Let's compute weights for each pair
    # w_j = exp(-beta * d_j) / sum_k exp(-beta * d_k)
    
    if len(sq_dists_finite) > 0:
        # Recompute exp terms for gradient
        d_min_grad = np.min(dists_offdiag[np.isfinite(dists_offdiag)])
        exp_terms_grad = np.exp(-beta * (dists_offdiag - d_min_grad))
        exp_terms_grad[~np.isfinite(exp_terms_grad)] = 0.0
        sum_exp = np.sum(exp_terms_grad)
        weights = exp_terms_grad / sum_exp  # (N, N) matrix
        
        # weights[i,j] corresponds to pair (i,j)
        # d(soft_min_pair)/d(c_i) = -sum_j weights[i,j] * (c_i - c_j) / (2 * d_ij)
        # where d_ij = ||c_i - c_j||
        
        # We need to be careful: weights is symmetric, and we're summing over all j != i
        # Let's compute the force on each center
        grad_pair = np.zeros_like(centers)
        
        # For each i, sum over j != i of weights[i,j] * (c_i - c_j) / (2 * ||c_i - c_j||)
        # Note: d_ij in the formula above is the full distance, and we have d_j = d_ij/2
        # So d(d_j)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
        
        # Vectorized: 
        # grad_pair[i] = -sum_j weights[i,j] * (c_i - c_j) / (2 * ||c_i - c_j||)
        # = -c_i * sum_j weights[i,j] / (2*d_ij) + sum_j weights[i,j] * c_j / (2*d_ij)
        
        # But weights[i,j] is defined for the soft-min, and we need to divide by d_ij
        # Let's define W[i,j] = weights[i,j] / d_ij for d_ij > 0
        safe_dists = np.where(dists_offdiag > 1e-12, dists_offdiag, 1.0)
        W = weights / safe_dists
        W[~np.isfinite(W)] = 0.0
        
        sum_W = np.sum(W, axis=1)
        weighted_centers = W @ centers
        
        grad_pair = -(centers * sum_W[:, np.newaxis] - weighted_centers)
        
        # The factor of 2 is already in dists_offdiag = dists/2, so safe_dists = dists/2
        # And we have d(d_j)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||) = (c_i - c_j) / (4 * d_j)
        # Wait, let me re-derive.
        # d_j = ||c_i - c_j|| / 2
        # d(d_j)/d(c_i) = (1/2) * (c_i - c_j) / ||c_i - c_j||
        # So grad_pair[i] = -sum_j weights[i,j] * (1/2) * (c_i - c_j) / ||c_i - c_j||
        # = -1/2 * sum_j weights[i,j] / ||c_i - c_j|| * (c_i - c_j)
        
        # Let me redefine: W2[i,j] = weights[i,j] / ||c_i - c_j||
        full_dists = np.sqrt(sq_dists_masked)
        safe_full_dists = np.where(full_dists > 1e-12, full_dists, 1.0)
        W2 = weights / safe_full_dists
        W2[~np.isfinite(W2)] = 0.0
        
        sum_W2 = np.sum(W2, axis=1)
        weighted_centers2 = W2 @ centers
        
        grad_pair = -0.5 * (centers * sum_W2[:, np.newaxis] - weighted_centers2)
    else:
        grad_pair = np.zeros_like(centers)
    
    # For wall part:
    # soft_min_wall = w_min - (1/beta) * log(sum_k exp(-beta * w_k))
    # d(soft_min_wall)/d(c_ik) = -sum_k exp(-beta*w_k)/sum_k' exp(-beta*w_k') * d(w_k)/d(c_ik)
    # w_k = 0.5 - |c_ik|, so d(w_k)/d(c_ik) = -sign(c_ik)
    # So d(soft_min_wall)/d(c_ik) = sum_k exp(-beta*w_k)/sum_k' exp(-beta*w_k') * sign(c_ik)
    
    exp_terms_wall_grad = np.exp(-beta * (walls - w_min))
    exp_terms_wall_grad[~np.isfinite(exp_terms_wall_grad)] = 0.0
    sum_exp_wall = np.sum(exp_terms_wall_grad)
    wall_weights = exp_terms_wall_grad / sum_exp_wall
    
    signs = np.sign(centers)
    signs[centers == 0] = 1.0
    grad_wall = wall_weights * signs
    
    total_grad = grad_pair + grad_wall
    return soft_min_radius, total_grad.flatten()

def _objective_and_grad(x, beta=50.0):
    """
    Objective: maximize soft-min radius = minimize -soft_min radius.
    Returns (value, gradient) where value = -soft_min_radius.
    """
    radius, grad = _soft_min_radius_and_grad(x, beta)
    return -radius, -grad

def _optimize_lbfgsb(centers, time_limit=20.0, beta=50.0):
    """
    Optimize centers to maximize soft-min radius using L-BFGS-B.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _objective_and_grad(x, beta)
        return val, grad
    
    try:
        result = minimize(
            obj, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        final_radius = _compute_radius(result.x)
        return result.x.reshape(N, 3), final_radius
    except Exception:
        return centers.copy(), _compute_radius(centers)

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _bcc_init(n):
    """Generate body-centered cubic lattice initial configuration."""
    m = int(np.ceil((n / 2) ** (1.0/3.0)))
    spacing = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = (ix + 0.5) * spacing - 0.5
                y = (iy + 0.5) * spacing - 0.5
                z = (iz + 0.5) * spacing - 0.5
                centers.append([x, y, z])
                if ix < m-1 and iy < m-1 and iz < m-1:
                    bx = (ix + 1.0) * spacing - 0.5
                    by = (iy + 1.0) * spacing - 0.5
                    bz = (iz + 1.0) * spacing - 0.5
                    centers.append([bx, by, bz])
    
    centers = np.array(centers)
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        grid = _grid_init(n)
        centers = np.vstack([centers, grid[len(centers):n]])
    
    return centers

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _high_precision_polish(centers, time_limit=10.0):
    """
    High-precision polish: iteratively adjust the most constrained pairs/walls
    to equalize all critical distances.
    """
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        min_sq_dist = np.min(sq_dists)
        min_pair_dist = np.sqrt(min_sq_dist)
        min_pair_idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
        
        min_wall_dist = np.min(0.5 - np.abs(centers))
        min_wall_idx = np.unravel_index(np.argmin(0.5 - np.abs(centers)), (N, 3))
        
        r_pair = min_pair_dist / 2.0
        r_wall = min_wall_dist
        r = min(r_pair, r_wall)
        
        # Find all pairs/walls within epsilon of the minimum
        eps = 1e-8
        tolerance = r + eps
        
        if r_pair < r_wall:
            # Pair is critical
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            # Push apart to equalize
            target_dist = 2.0 * r_wall  # If wall is the constraint, push to match wall
            if dist < target_dist:
                push = (target_dist - dist) * d / dist * 0.5
                centers[i] += push
                centers[j] -= push
        else:
            # Wall is critical
            i, k = min_wall_idx
            wall = 0.5 - np.abs(centers[i, k])
            if wall < r_pair:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                # Push inward to match pair distance
                target_wall = r_pair
                if wall < target_wall:
                    centers[i, k] += (target_wall - wall) * sign
        
        centers = np.clip(centers, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return centers

def _sgld_anneal(centers, time_limit=30.0):
    """
    Stochastic Gradient Langevin Dynamics with annealing.
    Add noise to escape local minima while optimizing.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    lr = 0.001
    noise_scale = 0.01
    beta = 30.0
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Anneal noise
        progress = iteration / 10000.0
        current_noise = noise_scale * (1.0 - 0.5 * progress)
        current_lr = lr * (1.0 - 0.3 * progress)
        
        # Compute gradient
        _, grad = _objective_and_grad(centers.flatten(), beta)
        grad = grad.reshape(N, 3)
        
        # SGLD step
        noise = np.random.normal(0, current_noise, size=centers.shape)
        centers = centers - current_lr * grad + noise
        centers = np.clip(centers, -0.5 + 1e-15, 0.5 - 1e-15)
        
        # Check improvement
        current_radius = _compute_radius(centers)
        if current_radius > best_radius + 1e-10:
            best_radius = current_radius
            best_centers = centers.copy()
    
    return best_centers, best_radius

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # 1. Generate initial configurations
    inits = []
    inits.append(_grid_init(N))
    inits.append(_bcc_init(N))
    for seed in range(8):
        inits.append(_random_init(N, seed=seed))
    
    best_centers = None
    best_radius = 0.0
    
    # 2. Evaluate inits
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # 3. Run L-BFGS-B optimization from multiple starts
    remaining_time = deadline - time.time()
    time_per_opt = min(25.0, remaining_time / 4.0)
    
    # Optimize from top 4 initial configurations
    inits_sorted = sorted(inits, key=lambda c: _compute_radius(c), reverse=True)
    
    for idx, init in enumerate(inits_sorted[:4]):
        if time.time() > deadline - 30.0:
            break
        
        time_left = deadline - time.time() - 30.0
        if time_left < 5.0:
            break
        
        opt_time = min(time_per_opt, time_left)
        new_centers, new_radius = _optimize_lbfgsb(init, time_limit=opt_time, beta=50.0)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
    
    # 4. SGLD annealing from best configuration
    if time.time() < deadline - 20.0:
        sgld_time = min(30.0, deadline - time.time() - 20.0)
        sgld_centers, sgld_radius = _sgld_anneal(best_centers, time_limit=sgld_time)
        
        if sgld_radius > best_radius:
            best_radius = sgld_radius
            best_centers = sgld_centers
    
    # 5. Final L-BFGS-B polish
    if time.time() < deadline - 10.0:
        polish_time = min(15.0, deadline - time.time() - 10.0)
        polished_centers, polished_radius = _optimize_lbfgsb(best_centers, time_limit=polish_time, beta=100.0)
        
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
    
    # 6. High-precision critical pair polish
    if time.time() < deadline - 5.0:
        hp_time = min(8.0, deadline - time.time() - 5.0)
        hp_centers = _high_precision_polish(best_centers, time_limit=hp_time)
        hp_radius = _compute_radius(hp_centers)
        
        if hp_radius > best_radius:
            best_radius = hp_radius
            best_centers = hp_centers
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.49999999, 0.49999999)
    
    return best_centers
# EVOLVE-BLOCK-END
