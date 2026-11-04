"""Optimized sphere packing in a unit cube using gradient descent with soft-min surrogate."""
import numpy as np
import time
import sys

N = 100
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum feasible radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check all centers are inside the cube (with strict interior)
    if not np.all(centers > -CUBE_HALF) or not np.all(centers < CUBE_HALF):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(centers + CUBE_HALF, CUBE_HALF - centers),
        CUBE_HALF
    )
    min_wall_dist = np.min(wall_dists)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)

def _soft_min_obj(centers, beta):
    """
    Compute a smooth surrogate for the negative of the minimum distance.
    We want to maximize the minimum of {pairwise distances, 2*wall distances}.
    
    Using soft-min: softmin(x) = -(1/beta) * log(sum(exp(-beta*x_i)))
    This is differentiable and approximates min(x) as beta -> inf.
    
    Returns (objective_value, gradient) where we minimize this.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances (distance from center to nearest wall, times 2 for symmetry with pairwise/2)
    # Actually, let's work with the actual radius constraints:
    # r = min( min_{i<j} dist_ij / 2, min_i wall_dist_i )
    # So we want to maximize this. Let's define:
    # values = [dist_ij / 2 for all i<j, wall_dist_i for all i]
    # And maximize the minimum of values.
    
    # Get upper triangle indices for pairwise
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0  # shape (n*(n-1)/2,)
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(centers + CUBE_HALF, CUBE_HALF - centers),
        CUBE_HALF
    ).min(axis=1)  # shape (n,)
    
    # Combine all values
    all_vals = np.concatenate([pair_dists, wall_dists])
    m = all_vals.shape[0]
    
    # Soft-min: -1/beta * log(sum(exp(-beta * val)))
    # For numerical stability, subtract max
    beta_val = beta
    neg_beta_vals = -beta_val * all_vals
    max_neg = np.max(neg_beta_vals)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_beta_vals - max_neg)))
    soft_min_val = -log_sum_exp / beta_val
    
    # Gradient with respect to centers
    # d(soft_min)/d(val_j) = exp(-beta*val_j) / sum(exp(-beta*val_k))
    weights = np.exp(neg_beta_vals - max_neg) / np.sum(np.exp(neg_beta_vals - max_neg))
    # soft_min = -1/beta * log(sum(exp(-beta*val)))
    # d(soft_min)/d(val_j) = (-1/beta) * (1/sum) * (-beta * exp(-beta*val_j))
    # = exp(-beta*val_j) / sum(exp(-beta*val_k)) = weights[j]
    # So gradient of soft_min w.r.t. val_j is weights[j]
    
    grad = np.zeros_like(centers)
    
    # Pairwise part: val_ij = dist_ij / 2
    # d(val_ij)/d(centers_i) = (centers_i - centers_j) / dist_ij
    # d(val_ij)/d(centers_j) = (centers_j - centers_i) / dist_ij
    # weight for pair (i,j) is weights[idx]
    
    # Reconstruct full distance matrix for gradient
    dists_safe = np.where(dists > 1e-15, dists, 1.0)
    
    # For each pair (i,j) with i<j, weight w_ij
    # grad_i += w_ij * (centers_i - centers_j) / dist_ij
    # grad_j += w_ij * (centers_j - centers_i) / dist_ij
    
    ii, jj = iu
    w_pairs = weights[:len(ii)]  # weights for pairwise
    diff_ij = centers[ii] - centers[jj]
    dist_ij = dists_safe[ii, jj]
    
    # grad contribution from pairs
    grad_pair = np.zeros_like(centers)
    np.add.at(grad_pair, ii, w_pairs[:, np.newaxis] * diff_ij / dist_ij[:, np.newaxis])
    np.add.at(grad_pair, jj, -w_pairs[:, np.newaxis] * diff_ij / dist_ij[:, np.newaxis])
    
    # Wall part: val_i = min over axes of wall distances for center i
    # Actually wall_dists[i] = min over k of min(centers[i,k]+0.5, 0.5-centers[i,k])
    # This is piecewise linear. The gradient is -1 or +1 depending on which constraint is active.
    # For soft-min, the gradient w.r.t. wall_dists[i] is weights[m - n + i]
    # Then d(wall_dists[i])/d(centers[i,k]) is the subgradient of the min.
    
    # For each center i, wall_dists[i] = min_k f_k(centers[i])
    # where f_k(c) = min(c_k + 0.5, 0.5 - c_k)
    # The active constraint is the one giving the minimum.
    
    w_walls = weights[len(ii):]  # weights for wall distances, shape (n,)
    
    # For each center i, compute which axis and which side is active
    for i in range(n):
        c = centers[i]
        # f_k for each axis
        f_left = c + CUBE_HALF   # distance to left wall
        f_right = CUBE_HALF - c  # distance to right wall
        # wall_dist = min over all 6 faces
        vals = np.concatenate([f_left, f_right])
        min_val = np.min(vals)
        # Gradient of wall_dist w.r.t. centers[i]
        # If left face active (f_left[k] is min), grad_k = 1
        # If right face active (f_right[k] is min), grad_k = -1
        active_left = (f_left == min_val)
        active_right = (f_right == min_val)
        grad_wall_i = np.zeros(3)
        grad_wall_i[active_left] += 1.0
        grad_wall_i[active_right] -= 1.0
        # Normalize if multiple active (shouldn't happen generically)
        grad[i] += w_walls[i] * grad_wall_i
    
    grad = grad_pair + grad
    return soft_min_val, grad

def _adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """One step of Adam optimizer. Updates m, v in place, returns new params."""
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads * grads
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params_new = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params_new, m, v

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple restarts with different initial configurations
    np.random.seed(42)
    
    # Initial config 1: Simple cubic grid (baseline)
    m_grid = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m_grid) + 0.5) / m_grid - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_configs = [grid[:N].copy()]
    
    # Initial config 2: Random with jitter
    for _ in range(3):
        # Place points in a slightly perturbed grid
        m_g = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        tks = (np.arange(m_g) + 0.5) / m_g - 0.5
        g = np.array([(x, y, z) for z in tks for y in tks for x in tks])[:N].copy()
        noise = np.random.randn(N, 3) * 0.02
        g = g + noise
        # Clip to stay inside
        g = np.clip(g, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
        init_configs.append(g)
    
    # Initial config 3: Pure random
    for _ in range(2):
        rand_pts = np.random.uniform(-CUBE_HALF + 0.02, CUBE_HALF - 0.02, size=(N, 3))
        init_configs.append(rand_pts)
    
    # Optimize each initial configuration
    for init_idx, init in enumerate(init_configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:  # Need at least 30s for remaining configs
            break
        
        centers = init.copy().flatten()  # 300-d vector
        n_params = centers.shape[0]
        
        # Adam state
        m_state = np.zeros(n_params)
        v_state = np.zeros(n_params)
        t = 0
        
        # Adaptive beta: start large (smoother), decrease over time (sharper)
        beta_start = 50.0
        beta_end = 500.0
        total_steps = 5000
        lr_start = 0.0005
        lr_end = 0.00005
        
        for step in range(total_steps):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            
            t += 1
            # Anneal beta and learning rate
            progress = step / total_steps
            beta = beta_start + (beta_end - beta_start) * progress
            lr = lr_start * (lr_end / lr_start) ** progress
            
            # Reshape to (N, 3)
            c = centers.reshape(N, 3)
            obj, grad = _soft_min_obj(c, beta)
            grad_flat = grad.flatten()
            
            # Clip gradient for stability
            grad_norm = np.linalg.norm(grad_flat)
            if grad_norm > 100.0:
                grad_flat = grad_flat * (100.0 / grad_norm)
            
            centers, m_state, v_state = _adam_step(centers, grad_flat, m_state, v_state, t, lr=lr)
            
            # Project back into cube
            c_new = centers.reshape(N, 3)
            c_new = np.clip(c_new, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
            centers = c_new.flatten()
            
            # Evaluate actual radius periodically
            if step % 500 == 0 or step == total_steps - 1:
                actual_r = _compute_radius(c_new)
                if actual_r > best_radius:
                    best_radius = actual_r
                    best_centers = c_new.copy()
        
        # Final evaluation for this restart
        c_final = centers.reshape(N, 3)
        c_final = np.clip(c_final, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        final_r = _compute_radius(c_final)
        if final_r > best_radius:
            best_radius = final_r
            best_centers = c_final.copy()
    
    # If no valid result found, fall back to grid
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m_grid = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m_grid) + 0.5) / m_grid - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    # Final safety: ensure valid
    best_centers = np.asarray(best_centers, dtype=np.float64)
    if best_centers.shape != (N, 3):
        m_grid = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m_grid) + 0.5) / m_grid - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
    
    return best_centers
