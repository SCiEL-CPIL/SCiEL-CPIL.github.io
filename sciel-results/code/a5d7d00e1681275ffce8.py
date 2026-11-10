# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    deadline = time.time() + 600  # Safety margin
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)
    
    def get_distances(points):
        # points: (n, d)
        # Returns array of shape (num_pairs,)
        diffs = points[i_idx] - points[j_idx]
        return np.linalg.norm(diffs, axis=1)

    def objective_and_grad(points, beta, lambda_scale):
        # points: (n, d)
        # We minimize L_max - L_min + lambda * (sum_sq_norm - 1)^2
        
        diffs = points[i_idx] - points[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Soft min and max
        # To avoid overflow/underflow in exp, shift by max/min
        # For L_min (approx log min dist):
        # min_dist = min(dists)
        # L_min = -1/beta * log(sum(exp(-beta * dists)))
        # Let's use log-sum-exp trick
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # L_max approx log(max_d)
        # log(sum(exp(beta * dists))) = beta * max_d + log(sum(exp(beta * (dists - max_d))))
        log_sum_exp_max = beta * max_d + np.log(np.sum(np.exp(beta * (dists - max_d))))
        L_max = log_sum_exp_max / beta
        
        # L_min approx log(min_d)
        # log(sum(exp(-beta * dists))) = -beta * min_d + log(sum(exp(-beta * (dists - min_d))))
        # Wait, exp(-beta * dists) = exp(-beta * min_d) * exp(-beta * (dists - min_d))
        # Sum = exp(-beta * min_d) * Sum(exp(-beta * (dists - min_d)))
        # Log Sum = -beta * min_d + log(Sum(...))
        log_sum_exp_min = -beta * min_d + np.log(np.sum(np.exp(-beta * (dists - min_d))))
        L_min = -log_sum_exp_min / beta
        
        obj = L_max - L_min
        
        # Scale penalty
        sum_sq_norm = np.sum(points**2)
        penalty = lambda_scale * (sum_sq_norm - 1.0)**2
        obj += penalty
        
        # Gradient
        # dL_max/dp_k
        # L_max = 1/beta * log(sum(exp(beta * d_ij)))
        # dL_max/dp_k = 1/beta * 1/sum(exp(beta * d_ij)) * sum( exp(beta * d_ij) * beta * d d_ij/dp_k )
        #             = sum( w_ij * d d_ij/dp_k ) where w_ij = exp(beta * d_ij) / sum(exp(beta * d_ij))
        
        exp_beta_dists = np.exp(beta * (dists - max_d)) # shifted
        sum_exp_max = np.sum(exp_beta_dists)
        w_max = exp_beta_dists / sum_exp_max
        
        # dL_min/dp_k
        # L_min = -1/beta * log(sum(exp(-beta * d_ij)))
        # dL_min/dp_k = -1/beta * 1/sum(exp(-beta * d_ij)) * sum( exp(-beta * d_ij) * (-beta) * d d_ij/dp_k )
        #             = sum( w'_ij * d d_ij/dp_k ) where w'_ij = exp(-beta * d_ij) / sum(exp(-beta * d_ij))
        
        exp_neg_beta_dists = np.exp(-beta * (dists - min_d)) # shifted
        sum_exp_min = np.sum(exp_neg_beta_dists)
        w_min = exp_neg_beta_dists / sum_exp_min
        
        # Gradient of objective w.r.t points
        grad = np.zeros_like(points)
        
        # Contribution from L_max
        # d d_ij / d p_k
        # For each pair (i, j), diff = p_i - p_j
        # d diff / d p_i = 1, d diff / d p_j = -1
        # d ||diff|| / d p_i = diff / ||diff||
        # d ||diff|| / d p_j = -diff / ||diff||
        
        # We need to accumulate gradients for each point
        # grad[p_i] += w_max_ij * (p_i - p_j) / d_ij
        # grad[p_j] -= w_max_ij * (p_i - p_j) / d_ij
        
        # Vectorized update is tricky with scatter, let's use a loop over pairs or np.add.at
        # np.add.at is efficient enough for 91 pairs
        
        # We need dists > 0 to avoid division by zero. 
        # If dists is 0, gradient is undefined, but probability is low. 
        # We can add a small epsilon to dists in the gradient calculation if necessary, 
        # or just let it be. If two points coincide, the objective is bad anyway.
        
        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        unit_vecs = diffs / safe_dists[:, np.newaxis]
        
        # grad_max contribution
        # For each pair, add w_max[i] * unit_vec[i] to p_i and subtract from p_j
        np.add.at(grad, i_idx, w_max[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -w_max[:, np.newaxis] * unit_vecs)
        
        # grad_min contribution (note the sign in L_min derivation)
        # dL_min/dp_k = sum( w'_ij * d d_ij/dp_k )
        # Wait, let's re-verify sign.
        # L_min = -1/beta log S_min. S_min = sum exp(-beta d).
        # d L_min = -1/beta (1/S_min) d S_min
        # d S_min = sum exp(-beta d) (-beta) d d
        # d L_min = -1/beta (1/S_min) (-beta) sum exp(-beta d) d d
        # d L_min = (1/S_min) sum exp(-beta d) d d
        # So yes, it is sum w'_ij d d_ij.
        
        np.add.at(grad, i_idx, w_min[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -w_min[:, np.newaxis] * unit_vecs)
        
        # Gradient of penalty: lambda * (sum_sq_norm - 1)^2
        # d/dp = 2 * lambda * (sum_sq_norm - 1) * 2 * p
        grad += 4.0 * lambda_scale * (sum_sq_norm - 1.0) * points
        
        return obj, grad

    best_points = None
    best_score = -np.inf
    
    # Try a few different betas and restarts
    betas = [10.0, 20.0, 50.0]
    lambda_scale = 1.0
    
    np.random.seed(42)
    
    num_restarts = 0
    max_restarts = 20
    
    for beta in betas:
        while time.time() < deadline - 10:
            # Random start on sphere
            points = np.random.randn(n, d)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            # Optimize
            try:
                res = scipy.optimize.minimize(
                    lambda x: objective_and_grad(x, beta, lambda_scale)[0],
                    points,
                    jac=lambda x: objective_and_grad(x, beta, lambda_scale)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                points_opt = res.x
                
                # Evaluate true metric
                dists = get_distances(points_opt)
                d_min = np.min(dists)
                d_max = np.max(dists)
                
                if d_max > 0:
                    score = (d_min / d_max) ** 2
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
            except Exception as e:
                pass
            
            num_restarts += 1
            if num_restarts >= max_restarts:
                break
                
    if best_points is None:
        # Fallback to random
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
