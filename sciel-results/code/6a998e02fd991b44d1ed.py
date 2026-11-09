"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 100

# EVOLVE-BLOCK-START

def _compute_radius_and_grad(c, beta):
    """
    Computes the soft-min radius and its gradient.
    c: (N, 3) array of centers.
    beta: inverse temperature for soft-min.
    Returns:
        r_soft: float, the soft-min radius value.
        grad: (N, 3) array, the gradient of r_soft with respect to c.
    """
    N = len(c)
    # Pairwise distances
    # dist[i, j] = ||c_i - c_j||
    # We need min_{i<j} dist[i,j] / 2
    # And min_i min_k (1/2 - |c_ik|)
    
    # Efficiently compute pairwise squared distances
    # D2[i, j] = ||c_i - c_j||^2
    # Using broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dist = np.sqrt(dist_sq)
    dist = dist[mask]
    
    # Boundary distances
    # dist_boundary[i, k] = 1/2 - |c_ik|
    # c is in [-0.5, 0.5]
    # We want min_k (0.5 - |c_ik|)
    # This is equivalent to min_k (0.5 - abs(c_ik))
    # Let's compute the vector of boundary distances for all points and axes
    # b_dist shape (N, 3)
    b_dist = 0.5 - np.abs(c)
    
    # Combine all distances into one vector
    # Note: The objective is min( min_pair_dist/2, min_boundary_dist )
    # Let d_pair = dist / 2.0
    # Let d_bound = b_dist
    # We want soft_min( concat(d_pair, d_bound) )
    
    d_pair = dist / 2.0
    d_bound = b_dist.flatten()
    
    # All distances
    d_all = np.concatenate([d_pair, d_bound])
    
    # Soft min: -1/beta * log( sum( exp(-beta * d) ) )
    # To avoid overflow/underflow, use log-sum-exp trick
    # max_d = np.max(d_all)
    # exp_terms = np.exp(-beta * (d_all - max_d))
    # log_sum = np.log(np.sum(exp_terms))
    # soft_min = -1/beta * (log_sum - beta * max_d) = max_d - 1/beta * log_sum
    
    # However, we also need the gradient.
    # Let S = sum_k exp(-beta * d_k)
    # f = -1/beta * log(S)
    # df/dc_i = -1/beta * (1/S) * sum_k d(exp(-beta d_k))/dc_i
    # d(exp(-beta d_k))/dc_i = -beta * exp(-beta d_k) * (dd_k/dc_i)
    # df/dc_i = -1/beta * (1/S) * sum_k (-beta exp(-beta d_k) dd_k/dc_i)
    # df/dc_i = (1/S) * sum_k exp(-beta d_k) dd_k/dc_i
    # Let w_k = exp(-beta d_k) / S
    # df/dc_i = sum_k w_k dd_k/dc_i
    
    # Compute weights
    # Scale d_all to prevent overflow
    min_d = np.min(d_all)
    # Shift by min_d so that exp(-beta*(d-min_d)) is <= 1
    shifted = d_all - min_d
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    # Gradient calculation
    # grad = sum_k w_k * grad(d_k)
    
    # 1. Pairwise part
    # d_k corresponds to dist[i,j]/2
    # grad_{c_i} (dist[i,j]/2) = (c_i - c_j) / dist[i,j]
    # grad_{c_j} (dist[i,j]/2) = (c_j - c_i) / dist[i,j]
    
    # We have flattened d_pair. We need to map indices back.
    # This is slow if done naively. Let's use a vectorized approach for the gradient.
    
    # Reconstruct full matrix of weights for pairs
    # W_pair[i,j] = weights[k] where k corresponds to (i,j)
    # It's easier to iterate or use sparse logic, but N=100 is small enough for some vectorization.
    
    # Let's build the gradient for pairs
    grad_pair = np.zeros_like(c)
    
    # Create a matrix of weights for the upper triangle
    # We need to map the flattened index to i, j
    # For N=100, N*(N-1)/2 = 4950 pairs.
    # We can create a meshgrid of indices
    i_idx, j_idx = np.where(mask)
    
    # weights_pair = weights[:len(d_pair)]
    w_pair = weights[:len(d_pair)]
    
    # Vectorized accumulation
    # For each pair (i, j) with weight w:
    # grad[i] += w * (c[i] - c[j]) / dist[i,j]
    # grad[j] += w * (c[j] - c[i]) / dist[i,j]
    
    # dist_matrix for pairs
    # We have dist vector corresponding to i_idx, j_idx
    # Avoid division by zero (should not happen if distinct)
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    
    # diff_vec = c[i] - c[j]
    # We can compute this vectorized
    # c_i = c[i_idx], c_j = c[j_idx]
    c_i = c[i_idx]
    c_j = c[j_idx]
    diff_vec = c_i - c_j
    
    # unit vectors
    unit_vecs = diff_vec / safe_dist[:, np.newaxis]
    
    # weighted unit vectors
    w_vecs = w_pair[:, np.newaxis] * unit_vecs
    
    # Accumulate to grad_pair
    # np.add.at
    np.add.at(grad_pair, i_idx, w_vecs)
    np.add.at(grad_pair, j_idx, -w_vecs)
    
    # 2. Boundary part
    # d_bound[i, k] = 0.5 - |c_ik|
    # grad_{c_ik} d_bound = -sign(c_ik)
    # w_bound = weights[len(d_pair):]
    w_bound = weights[len(d_pair):]
    
    # Reshape w_bound to (N, 3)
    w_bound_mat = w_bound.reshape(N, 3)
    
    # grad_bound[i, k] = w_bound[i, k] * (-sign(c[i, k]))
    # sign(0) is 0, which is fine (subgradient 0)
    signs = np.sign(c)
    grad_bound = -w_bound_mat * signs
    
    # Total gradient
    grad = grad_pair + grad_bound
    
    # Value
    # soft_min = min_d - 1/beta * log(S)
    # Wait, the shift was min_d.
    # S_shifted = sum exp(-beta (d - min_d))
    # log(S) = log(S_shifted) - beta * min_d
    # f = -1/beta (log(S_shifted) - beta min_d) = min_d - 1/beta log(S_shifted)
    r_soft = min_d - (1.0 / beta) * np.log(S)
    
    return r_soft, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initialization: Jittered FCC lattice or similar
    # Let's use a grid with random jitter to break symmetry
    # m = 5 gives 125 points. We take 100.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to take first 100
    np.random.seed(42)
    idx = np.random.permutation(len(grid))
    c = grid[idx[:N]].copy()
    
    # Add small jitter
    c += np.random.normal(0, 0.01, c.shape)
    
    # Ensure inside cube
    c = np.clip(c, -0.49, 0.49)
    
    best_c = c.copy()
    best_score = _compute_radius_and_grad(c, 100.0)[0] # Approximate initial score
    
    # Optimization parameters
    # Start with low beta (smooth) and increase
    beta_start = 5.0
    beta_end = 200.0
    n_steps = 5000
    
    # We will run a loop, checking time
    # Adaptive step size
    step_size = 0.01
    
    current_beta = beta_start
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Update beta
        # Linear or exponential schedule
        progress = step / n_steps
        # Exponential increase is often better
        current_beta = beta_start * (beta_end / beta_start) ** progress
        
        r, grad = _compute_radius_and_grad(c, current_beta)
        
        # Normalize gradient for stable step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad_normed = grad / grad_norm
        else:
            grad_normed = grad
            
        # Step
        # Heuristic step size: smaller as beta increases
        # step = step_size * (1.0 / (1.0 + current_beta/10.0))
        # Or just fixed small step
        # Let's try adaptive step based on improvement
        
        c_new = c + step_size * grad_normed
        
        # Check if new position is valid (inside cube)
        # If it goes outside, we should reflect or clip, but clipping might break gradient.
        # For now, clip to [-0.5, 0.5] with small epsilon
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        # Compute new score (using high beta for evaluation of progress)
        # Actually, just check if soft-min improved
        r_new, _ = _compute_radius_and_grad(c_new, current_beta)
        
        if r_new > r:
            c = c_new
            # Maybe increase step size slightly?
            # step_size = min(step_size * 1.05, 0.05)
        else:
            # Decrease step size
            step_size = max(step_size * 0.95, 0.001)
            # Try a smaller step or random perturbation if stuck
            if step % 100 == 0 and step > 100:
                 # Add noise
                 c_new = c + np.random.normal(0, 0.005, c.shape)
                 c_new = np.clip(c_new, -0.4999, 0.4999)
                 r_new, _ = _compute_radius_and_grad(c_new, current_beta)
                 if r_new > r:
                     c = c_new
        
        # Track best
        # Use a fixed high beta for "best" tracking to ensure it's a good hard-min approximation
        if step % 10 == 0:
            r_check, _ = _compute_radius_and_grad(c, 100.0)
            if r_check > best_score:
                best_score = r_check
                best_c = c.copy()
        
        # Time check
        if time.time() > deadline - 5.0:
             break
    
    # Final polish: use the best_c found
    # Do a few more steps with very high beta on best_c
    c = best_c
    beta_final = 500.0
    step_size = 0.001
    
    # Determine remaining time
    remaining = deadline - time.time()
    if remaining > 1.0:
        # Estimate steps we can do
        # Each step takes ~0.5ms? Let's assume 1ms.
        max_polish_steps = int(remaining * 0.8 * 1000) # 80% of time
        max_polish_steps = min(max_polish_steps, 2000)
        
        for step in range(max_polish_steps):
            if time.time() > deadline - 0.5:
                break
            r, grad = _compute_radius_and_grad(c, beta_final)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            grad_normed = grad / grad_norm
            c_new = c + step_size * grad_normed
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            # Check improvement
            r_new, _ = _compute_radius_and_grad(c_new, beta_final)
            if r_new > r:
                c = c_new
            else:
                # Try smaller step
                step_size *= 0.9
                if step_size < 1e-6:
                    break
    
    # Final validation and return
    # Ensure strictly inside
    c = np.clip(c, -0.499999, 0.499999)
    
    return c

# EVOLVE-BLOCK-END
