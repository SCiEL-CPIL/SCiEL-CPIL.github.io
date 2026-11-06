"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_radius(centers):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Wall distances
    # Distance to closest wall for each center
    # min(0.5 - c_x, 0.5 + c_x, ...) = 0.5 - max(|c_x|, |c_y|, |c_z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall = np.min(wall_dists)
    
    if n < 2:
        return min_wall
        
    # Pairwise distances
    # We need the minimum distance between any pair
    # Using scipy.spatial.distance.pdist for efficiency? 
    # Or vectorized numpy.
    # For N=50, N^2 is 2500, which is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_d2 = np.min(d2)
    min_pair_dist = np.sqrt(min_pair_d2)
    
    return min(min_pair_dist / 2.0, min_wall)

def _soft_min_radius_and_grad(centers, beta):
    """
    Compute the soft-min radius and its gradient.
    Objective to minimize: -soft_min_radius
    """
    n = len(centers)
    
    # 1. Compute pairwise distances and their gradients contributions
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    d2_safe = np.where(d2 < 1e-12, 1e-12, d2)
    dist = np.sqrt(d2_safe)
    
    # We only consider i < j
    iu = np.triu_indices(n, k=1)
    d_pair = dist[iu] / 2.0 # The radius constraint is dist/2 >= r
    
    # 2. Compute wall distances
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    d_wall = 0.5 - max_abs
    
    # Combine all constraints
    # We want to maximize min( d_pair, d_wall )
    # Soft min: m_beta = -1/beta * log( sum( exp(-beta * d_k) ) )
    all_d = np.concatenate([d_pair, d_wall])
    
    if len(all_d) == 0:
        return 0.0, np.zeros_like(centers)
        
    # Stable log-sum-exp
    d_min = np.min(all_d)
    shifted = all_d - d_min
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    log_sum = np.log(sum_exp)
    
    soft_min_val = d_min - (1.0 / beta) * log_sum
    
    # Gradient
    # d(soft_min)/d(c) = sum_k w_k * d(d_k)/d(c)
    # where w_k = exp(-beta * d_k) / sum_exp
    weights = exp_term / sum_exp
    
    w_pair = weights[:len(d_pair)]
    w_wall = weights[len(d_pair):]
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
    # d(d_ij)/d(c_j) = 0.5 * (c_j - c_i) / |c_i - c_j|
    # Note: d_pair in all_d is |c_i - c_j| / 2
    # So derivative of d_pair w.r.t c_i is 0.5 * (c_i - c_j) / dist_ij
    # Wait, d_pair = 0.5 * dist.
    # d(d_pair)/d(c_i) = 0.5 * (c_i - c_j) / dist.
    
    diff_iu = diff[iu]
    dist_iu = dist[iu]
    # Unit vectors
    u_iu = diff_iu / dist_iu[:, np.newaxis]
    
    # Weighted unit vectors
    # grad[i] += w_pair[k] * 0.5 * u_k
    # grad[j] -= w_pair[k] * 0.5 * u_k
    w_u = u_iu * (w_pair[:, np.newaxis] * 0.5)
    
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, w_u)
    np.add.at(grad, j_idx, -w_u)
    
    # Wall gradient
    # d(d_wall_i)/d(c_i) = - d(max_abs_i)/d(c_i)
    # max_abs_i = max(|c_ix|, |c_iy|, |c_iz|)
    # Subgradient is sign(c_ik) if |c_ik| is the max, 0 otherwise.
    # For smoothness, we can use a soft-max on the absolute values?
    # Or just use the hard subgradient. L-BFGS-B can handle it if we are careful, 
    # but soft is better.
    # Let's use a soft-max approximation for the max to make it differentiable.
    # max(x) ~ 1/beta_inner * log(sum(exp(beta_inner * x)))
    # Let's use beta_inner = beta.
    
    # Recalculate wall part with soft-max for smoothness?
    # If we change the value of d_wall, we must change the weights too.
    # The current code uses hard max for d_wall values but hard subgradient for grad.
    # This is inconsistent if we want a true gradient.
    # However, the set of points where |cx|=|cy| is measure zero.
    # Let's stick to hard max for value, and handle ties by averaging or picking one.
    # Actually, let's just use the subgradient of the hard max.
    # If |cx| > |cy| and |cx| > |cz|, grad_x = sign(cx), grad_y=0, grad_z=0.
    
    max_abs_vals = np.max(abs_c, axis=1)
    # Find indices of max
    # If there are ties, we can distribute the gradient.
    # For simplicity and speed, let's just pick the first max index.
    # This introduces a small non-smoothness but usually works.
    
    # Better: Use the soft-max gradient derived earlier?
    # Let's try to implement the soft-max gradient for the wall part to be fully smooth.
    # d_wall_soft = 0.5 - soft_max(abs_c)
    # soft_max(x) = 1/beta * log(sum(exp(beta*x)))
    # d(soft_max)/d(x_k) = exp(beta*x_k) / sum(exp(beta*x_j))
    
    # Let's re-compute d_wall and weights using soft-max.
    # This ensures consistency between value and gradient.
    
    # Soft max of abs_c
    # To avoid overflow, subtract max
    max_abs_hard = np.max(abs_c, axis=1, keepdims=True)
    exp_abs = np.exp(beta * (abs_c - max_abs_hard))
    sum_exp_abs = np.sum(exp_abs, axis=1, keepdims=True)
    w_abs = exp_abs / sum_exp_abs # Shape (n, 3)
    
    # Soft max value
    soft_max_val = max_abs_hard[:, 0] + (1.0 / beta) * np.log(sum_exp_abs[:, 0])
    
    # New d_wall
    d_wall_new = 0.5 - soft_max_val
    
    # Recompute all_d and weights with d_wall_new
    all_d_new = np.concatenate([d_pair, d_wall_new])
    d_min_new = np.min(all_d_new)
    shifted_new = all_d_new - d_min_new
    exp_term_new = np.exp(-beta * shifted_new)
    sum_exp_new = np.sum(exp_term_new)
    log_sum_new = np.log(sum_exp_new)
    
    soft_min_val_new = d_min_new - (1.0 / beta) * log_sum_new
    
    weights_new = exp_term_new / sum_exp_new
    w_pair_new = weights_new[:len(d_pair)]
    w_wall_new = weights_new[len(d_pair):]
    
    # Recompute grad
    grad = np.zeros_like(centers)
    
    # Pairwise (same as before, but with new weights)
    w_u_new = u_iu * (w_pair_new[:, np.newaxis] * 0.5)
    np.add.at(grad, i_idx, w_u_new)
    np.add.at(grad, j_idx, -w_u_new)
    
    # Wall
    # d(d_wall_new)/d(c_i) = - d(soft_max(abs_c_i))/d(c_i)
    # d(soft_max)/d(c_ik) = w_abs_ik * sign(c_ik)
    # So d(d_wall)/d(c_ik) = - w_abs_ik * sign(c_ik)
    
    sign_c = np.sign(centers)
    # grad += w_wall_new[:, np.newaxis] * (- w_abs * sign_c)
    wall_grad = -w_wall_new[:, np.newaxis] * w_abs * sign_c
    grad += wall_grad
    
    return soft_min_val_new, grad

def _objective(x, beta):
    centers = x.reshape(N, 3)
    val, _ = _soft_min_radius_and_grad(centers, beta)
    return -val

def _gradient(x, beta):
    centers = x.reshape(N, 3)
    _, grad = _soft_min_radius_and_grad(centers, beta)
    return -grad.flatten()

def _init_bcc():
    # 3x3x3 grid of BCC cells
    # Each cell has 2 points: corner and center
    pts = []
    step = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * step
                oy = -0.5 + j * step
                oz = -0.5 + k * step
                pts.append([ox, oy, oz])
                pts.append([ox + step/2, oy + step/2, oz + step/2])
    pts = np.array(pts)
    # Add small noise to break symmetry
    pts += np.random.normal(0, 0.01, pts.shape)
    return pts[:N]

def _init_grid():
    # 4x4x4 grid
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid += np.random.normal(0, 0.01, grid.shape)
    return grid[:N]

def _init_random():
    pts = np.random.uniform(-0.45, 0.45, (N, 3))
    return pts

def _optimize_local(centers, time_budget, beta_start=100, beta_end=1000, n_phases=5):
    start = time.time()
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    betas = np.linspace(beta_start, beta_end, n_phases)
    
    for beta in betas:
        if time.time() - start > time_budget:
            break
            
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = opt.minimize(
                _objective, x0,
                jac=lambda x: _gradient(x, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.status == 0:
                new_centers = result.x.reshape(N, 3)
                if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5 + 1e-9):
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    r = _compute_radius(new_centers)
                    if r > best_r:
                        best_r = r
                        best_centers = new_centers.copy()
                    centers = new_centers.copy()
        except Exception:
            pass
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_centers = None
    best_r = -np.inf
    
    initializers = [
        _init_bcc,
        _init_grid,
        _init_random,
        _init_bcc, # Second BCC with different noise
        _init_random # Second Random
    ]
    
    # Allocate time per initialization
    # We have ~140s. Let's give each init ~20s.
    # But we can do better by checking time remaining.
    
    for init_func in initializers:
        if time.time() - start_time > deadline - 5.0:
            break
            
        centers = init_func()
        # Ensure valid start
        centers = np.clip(centers, -0.49, 0.49)
        
        # Give this run a budget.
        # If we have 5 inits, ~28s each.
        budget = min(28.0, deadline - 5.0 - (time.time() - start_time))
        if budget < 1.0:
            break
            
        c, r = _optimize_local(centers, budget)
        
        if r > best_r:
            best_r = r
            best_centers = c.copy()
            
    # If we still have time, do a final polish on the best solution
    if best_centers is not None and time.time() - start_time < deadline - 2.0:
        budget = deadline - 2.0 - (time.time() - start_time)
        if budget > 1.0:
            # Use a higher beta for final polish
            c, r = _optimize_local(best_centers, budget, beta_start=1000, beta_end=5000, n_phases=3)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
                
    # Fallback
    if best_centers is None:
        best_centers = _init_grid()
        
    return best_centers
# EVOLVE-BLOCK-END
