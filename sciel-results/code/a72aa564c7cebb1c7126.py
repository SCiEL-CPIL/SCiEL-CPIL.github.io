import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Start with a reasonable grid-like initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    np.random.seed(42)
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    # We'll use a simulated annealing approach with careful handling
    # to maximize the minimum distance (packing radius)
    
    def compute_radius(centers):
        """Compute the packing radius (half the minimum distance)."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Wall distances
        dist_x = 0.5 - np.abs(centers[:, 0])
        dist_y = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.concatenate([dist_x, dist_y]))
        
        return min(min_pair, min_wall) / 2.0
    
    def compute_min_dists(centers):
        """Return the minimum pair distance and minimum wall distance."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        dist_x = 0.5 - np.abs(centers[:, 0])
        dist_y = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.concatenate([dist_x, dist_y]))
        
        return min_pair, min_wall
    
    # Simulated annealing with momentum
    current = centers.copy()
    best = current.copy()
    best_r = compute_radius(best)
    
    T_start = 0.02
    T_end = 0.0001
    n_iter = 3000
    
    momentum = np.zeros_like(current)
    mu = 0.9  # momentum coefficient
    
    for it in range(n_iter):
        t = it / (n_iter - 1)
        T = T_start * (T_end / T_start) ** t
        
        # Generate perturbation
        noise = np.random.normal(0, T, current.shape)
        trial = current + noise
        trial = np.clip(trial, -0.5 + 1e-9, 0.5 - 1e-9)
        
        r_trial = compute_radius(trial)
        r_current = compute_radius(current)
        
        # Accept if better, or with probability exp((r_trial - r_current) / T)
        if r_trial >= r_current:
            accept = True
        else:
            delta_r = r_trial - r_current
            accept = np.random.random() < np.exp(delta_r / T)
        
        if accept:
            # Use momentum for smoother updates
            momentum = mu * momentum + (1 - mu) * (trial - current)
            current = trial
            if r_trial > best_r:
                best = current.copy()
                best_r = r_trial
        else:
            # Don't update current, but still dampen momentum
            momentum = mu * momentum
    
    # Local refinement using scipy minimize on a soft objective
    # Maximize log-sum-exp of distances to push up the minimum
    
    def neg_log_sum_exp(obj, beta=100.0):
        """Negative of log-sum-exp of distances, which we minimize."""
        c = obj.reshape(N, 2)
        # Clip to valid range
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists[~np.eye(N, dtype=bool)]
        
        # Wall distances
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_y = 0.5 - np.abs(c[:, 1])
        wall_dists = np.concatenate([dist_x, dist_y])
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Log-sum-exp
        m = np.max(all_dists)
        lse = m + np.log(np.sum(np.exp(all_dists - m)))
        return -lse
    
    def neg_log_sum_exp_grad(obj, beta=100.0):
        """Gradient of the negative log-sum-exp."""
        c = obj.reshape(N, 2)
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists[~np.eye(N, dtype=bool)]
        
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_y = 0.5 - np.abs(c[:, 1])
        wall_dists = np.concatenate([dist_x, dist_y])
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        m = np.max(all_dists)
        weights = np.exp(all_dists - m)
        sum_w = np.sum(weights)
        
        # Pairwise gradients
        # Reconstruct full distance matrix
        dists_mat = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists_mat, np.inf)
        
        # Weights matrix
        w_mat = np.exp(dists_mat - m)
        w_mat[np.eye(N, dtype=bool)] = 0.0
        
        # Gradient of log-sum-exp w.r.t. c_i
        # For pairs: d/d(c_i) log(sum exp(d_ij)) = sum_j w_ij * (c_i - c_j)/d_ij / sum_w
        diff_mat = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_safe = np.where(dists_mat > 1e-10, dists_mat, 1.0)
        unit_vecs = diff_mat / dists_safe[:, :, np.newaxis]
        unit_vecs[np.eye(N, dtype=bool)] = 0.0
        
        grad_pairs = np.sum(w_mat[:, :, np.newaxis] * unit_vecs, axis=1) / sum_w
        
        # Wall gradients
        grad_walls = np.zeros_like(c)
        # x-component
        sign_x = np.where(c[:, 0] < 0, -1.0, 1.0)  # d/dx of (0.5-|x|) = -sign(x)
        w_x = np.exp(dist_x - m)
        grad_walls[:, 0] += sign_x * w_x / sum_w
        
        # y-component
        sign_y = np.where(c[:, 1] < 0, -1.0, 1.0)
        w_y = np.exp(dist_y - m)
        grad_walls[:, 1] += sign_y * w_y / sum_w
        
        grad = grad_pairs + grad_walls
        return -grad.flatten()  # Negative because we minimize
    
    # Refine with L-BFGS-B
    x0 = best.flatten()
    result = minimize(neg_log_sum_exp, x0, method='L-BFGS-B', 
                      jac=neg_log_sum_exp_grad,
                      bounds=[(-0.5, 0.5)] * (2 * N),
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    refined = result.x.reshape(N, 2)
    refined = np.clip(refined, -0.5 + 1e-9, 0.5 - 1e-9)
    
    r_refined = compute_radius(refined)
    if r_refined > best_r:
        best = refined
        best_r = r_refined
    
    return best
