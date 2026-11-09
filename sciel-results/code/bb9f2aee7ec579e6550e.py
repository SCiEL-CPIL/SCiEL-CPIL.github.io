import numpy as np
import time

N = 100
SEED_RADIUS = 0.1113823475125

def _compute_score(centers):
    """Compute the radius score for a given set of centers."""
    # Centers must be in [-0.5, 0.5]^3
    # Distance to nearest wall
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use efficient computation
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    min_pair = np.min(d)
    
    r = min(min_pair / 2.0, min_wall)
    return r

def _soft_min_score(centers, alpha=100.0):
    """
    Compute a smooth approximation of the score.
    score = min(min_pair/2, min_wall)
    We maximize this.
    
    Using soft-min:
    S = (1/alpha) * log( sum( exp(-alpha * (min_pair/2)) + exp(-alpha * min_wall) ) )
    This is a smooth function of centers.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # To avoid O(N^2) in gradient computation if we were doing analytic, 
    # but here we just need the value for the optimizer.
    # However, for gradient ascent, we need to be careful.
    # Let's use a simpler approach: 
    # We want to maximize r.
    # Let's define an energy function E = -r_smooth.
    
    # Actually, let's just use the raw score but with a better optimizer?
    # No, the lesson says "soft-min gradient ascent".
    
    # Let's compute the soft min of the constraints.
    # Constraint 1: min_pair / 2 >= r
    # Constraint 2: min_wall >= r
    
    # We want to maximize r such that all constraints hold.
    # Equivalent to minimizing the "violation".
    
    # Let's use the soft-min of the two limiting values: d_min_pair/2 and d_min_wall.
    # S = softmin(d_min_pair/2, d_min_wall)
    
    # To make it differentiable w.r.t centers, we need the softmin of ALL pairwise distances and wall distances?
    # No, min_pair is the min of all pairs. min_wall is the min of all walls.
    # min(a, b) is not differentiable at a=b.
    
    # A common trick: 
    # r = softmin( [d_ij/2 for all i<j], [w_i for all i] )
    # This is a smooth function of centers.
    
    # Let's compute the vector of all constraints.
    # This is large: ~5000 pairs + 300 walls.
    
    from scipy.spatial.distance import pdist
    
    # Pairwise distances / 2
    d_pairs = pdist(centers) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    # Flatten wall distances
    w_flat = wall_dists.flatten()
    
    # Combine all constraints into one vector
    constraints = np.concatenate([d_pairs, w_flat])
    
    # Soft min
    # m = (1/alpha) * log( sum( exp(-alpha * c_i) ) )
    # To avoid overflow/underflow, use log-sum-exp trick
    min_c = np.min(constraints)
    # If min_c is very large, exp(-alpha * c) might underflow.
    # But we are maximizing r, so we expect c to be around 0.1.
    # alpha=100 -> exp(-10) ~ 4e-5. Safe.
    
    # For numerical stability:
    # logsumexp(-alpha * c) = -alpha * min_c + log(sum(exp(-alpha * (c - min_c))))
    shifted = -alpha * (constraints - min_c)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = min_c - log_sum / alpha
    
    return soft_min_val

def _grad_soft_min_score(centers, alpha=100.0):
    """
    Compute the gradient of the soft-min score w.r.t centers.
    We want to ascend this score.
    
    S = softmin(C), where C is the vector of constraints.
    dS/dx = sum_i (w_i * dC_i/dx), where w_i = exp(-alpha * C_i) / sum_j exp(-alpha * C_j)
    """
    from scipy.spatial.distance import pdist
    
    # Compute constraints
    d_pairs = pdist(centers) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    w_flat = wall_dists.flatten()
    
    constraints = np.concatenate([d_pairs, w_flat])
    
    # Weights
    min_c = np.min(constraints)
    shifted = -alpha * (constraints - min_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Gradient of softmin w.r.t constraints:
    # dS/dC_i = -weights[i]
    # So dS/dx = sum_i (-weights[i]) * dC_i/dx
    
    grad = np.zeros_like(centers)
    
    # 1. Pairwise gradients
    # C_i = |c_a - c_b| / 2
    # dC_i/dc_a = (c_a - c_b) / (2 * |c_a - c_b|)
    # dC_i/dc_b = (c_b - c_a) / (2 * |c_a - c_b|)
    
    # We need to map the condensed distance vector indices back to pairs.
    # This is O(N^2) and doing it in Python loops is slow.
    # Let's vectorize.
    
    # Compute all pairwise differences
    # diff[i, j, k] = c[i, k] - c[j, k]
    # This is N x N x 3. For N=100, 100*100*3 = 30,000 floats. Fine.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    # Unit vectors
    unit_vecs = diff / dists_safe[:, :, np.newaxis]
    
    # We only need i < j.
    # The condensed vector d_pairs corresponds to i < j.
    # Let's create a mask for i < j.
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Get the weights for the pairs
    # The weights vector `weights` has length N_pairs + N_walls.
    # The first N_pairs elements correspond to the pairs in pdist order.
    # pdist order is (0,1), (0,2), ..., (0,N-1), (1,2), ...
    # We need to map these weights to the (i, j) indices.
    
    # Instead of mapping, let's just compute the force for all i!=j and divide by 2?
    # No, pdist has specific order.
    
    # Let's reconstruct the pairwise force vector directly using the weights.
    # We need to iterate or use advanced indexing to scatter the weights.
    
    # Let's create an array of indices for i < j
    i_idx, j_idx = np.where(mask)
    
    # Get the corresponding weights
    pair_weights = weights[:len(i_idx)]
    
    # For each pair (i, j), the force on i is -pair_weight * unit_vec(i->j)
    # The force on j is -pair_weight * unit_vec(j->i) = +pair_weight * unit_vec(i->j)
    
    # dS/dx = sum -w_k * dC_k/dx
    # dC_k/dc_i = unit_vec(i, j)
    # dC_k/dc_j = unit_vec(j, i) = -unit_vec(i, j)
    
    # So contribution to grad[i] -= w_k * unit_vec(i, j)
    # Contribution to grad[j] -= w_k * (-unit_vec(i, j)) = + w_k * unit_vec(i, j)
    
    # Let's accumulate using np.add.at
    u_vecs = unit_vecs[i_idx, j_idx] # Shape (M, 3)
    
    # grad[i] -= w * u
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * u_vecs)
    # grad[j] += w * u
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * u_vecs)
    
    # 2. Wall gradients
    # C_i = 0.5 - |c_i|
    # dC_i/dc_i = -sign(c_i)
    # dS/dx = sum -w_wall * dC/dx = sum -w_wall * (-sign(c_i)) = w_wall * sign(c_i)
    
    wall_weights = weights[len(i_idx):]
    
    # wall_dists is (N, 3). Flatten order is C-order: (0,0), (0,1), (0,2), (1,0)...
    # So wall_weights[k] corresponds to center[k//3], axis[k%3]
    
    # sign(c)
    signs = np.sign(centers)
    # If c is 0, sign is 0. Gradient is 0. This is fine, softmin will handle it.
    
    # Reshape wall_weights to (N, 3)
    ww = wall_weights.reshape(N, 3)
    
    # grad += ww * signs
    grad += ww * signs
    
    return grad

def construct():
    """Construct centers of 100 equal spheres in the unit cube [-0.5, 0.5]^3."""
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s
    
    # Initialization: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_score = _compute_score(best_centers)
    
    # Optimization: Gradient Ascent with Soft-Min
    # We want to maximize S(centers).
    # Use Adam or simple gradient descent on -S.
    
    # Let's implement a simple adaptive gradient ascent (like Adam)
    # or just use scipy.optimize.minimize with BFGS providing our gradient?
    # BFGS might be better for local refinement.
    
    # Let's try BFGS with our custom gradient.
    
    from scipy.optimize import minimize
    
    def neg_score(x):
        c = x.reshape(N, 3)
        return -_soft_min_score(c, alpha=50.0)
        
    def neg_grad(x):
        c = x.reshape(N, 3)
        return -_grad_soft_min_score(c, alpha=50.0)
    
    # Try multiple restarts
    rng = np.random.default_rng(42)
    
    attempts = [best_centers.copy()]
    
    # Add perturbed grid
    perturbed = best_centers.copy()
    perturbed += rng.normal(0, 0.02, size=perturbed.shape)
    perturbed = np.clip(perturbed, -0.49, 0.49)
    attempts.append(perturbed)
    
    # Add random
    random_init = rng.uniform(-0.45, 0.45, size=(N, 3))
    attempts.append(random_init)
    
    for i, init in enumerate(attempts):
        if time.time() > deadline - 30:
            break
            
        try:
            x0 = init.flatten()
            # BFGS with custom gradient
            res = minimize(
                neg_score, 
                x0, 
                jac=neg_grad, 
                method='BFGS', 
                options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
            )
            
            opt_centers = res.x.reshape(N, 3)
            # Project to cube
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            
            # Evaluate with HARD score
            hard_score = _compute_score(opt_centers)
            
            if hard_score > best_score:
                best_score = hard_score
                best_centers = opt_centers.copy()
                print(f"Attempt {i}: improved to {best_score:.10f}")
                
        except Exception as e:
            print(f"Attempt {i} failed: {e}")
            continue
    
    # Final polish with higher alpha for sharper min
    if time.time() < deadline - 10:
        try:
            def neg_score_sharp(x):
                c = x.reshape(N, 3)
                return -_soft_min_score(c, alpha=200.0)
                
            def neg_grad_sharp(x):
                c = x.reshape(N, 3)
                return -_grad_soft_min_score(c, alpha=200.0)
                
            res = minimize(
                neg_score_sharp, 
                best_centers.flatten(), 
                jac=neg_grad_sharp, 
                method='BFGS', 
                options={'maxiter': 200, 'gtol': 1e-10, 'disp': False}
            )
            opt_centers = res.x.reshape(N, 3)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            hard_score = _compute_score(opt_centers)
            
            if hard_score > best_score:
                best_score = hard_score
                best_centers = opt_centers.copy()
                print(f"Polish: improved to {best_score:.10f}")
        except Exception as e:
            print(f"Polish failed: {e}")
    
    # Safety check
    if not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_centers
