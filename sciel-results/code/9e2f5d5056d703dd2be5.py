# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses gradient ascent on a smooth approximation of the log-ratio of min/max distances, 
    constrained to the unit sphere.
    """
    n = 14
    d = 3
    best_score = 0.0
    best_points = None
    
    # Precompute mapping from condensed index to (i, j) pairs
    # pdist order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    pair_indices = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    
    n_restarts = 200
    n_iterations = 2000
    
    for r in range(n_restarts):
        # Initialize points on the unit sphere
        np.random.seed(r)
        
        # Start with random points on sphere
        points = np.random.randn(n, d)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Momentum for gradient ascent
        velocity = np.zeros_like(points)
        mu = 0.9 # Momentum factor
        
        # Beta for soft min/max (inverse temperature)
        # Start with low beta (smooth) and increase
        beta_start = 5.0
        beta_end = 100.0
        
        current_score = 0.0
        alpha = 0.05 # Initial step size
        
        for i in range(n_iterations):
            # Compute pairwise distances
            dists = pdist(points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max == 0:
                break
            
            score = d_min / d_max
            
            if score > current_score:
                current_score = score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            
            # Anneal beta
            # Linear interpolation from beta_start to beta_end
            progress = i / n_iterations
            beta = beta_start + (beta_end - beta_start) * progress
            
            # Compute Soft Min and Soft Max gradients
            # softmin(D) = -1/beta * log(sum(exp(-beta * D)))
            # d(softmin)/dD_k = -exp(-beta * D_k) / sum(exp(-beta * D_j))
            # softmax(D) = 1/beta * log(sum(exp(beta * D)))
            # d(softmax)/dD_k = exp(beta * D_k) / sum(exp(beta * D_j))
            
            # To avoid overflow/underflow, we can shift D
            # But for n=14, D is small (distances on unit sphere <= 2), beta <= 100.
            # exp(-100*2) is tiny, exp(100*2) is huge.
            # We should use log-sum-exp trick or just be careful.
            # Actually, for gradient of log(softmin) - log(softmax):
            # grad_log_softmin_k = -exp(-beta*D_k) / sum(exp(-beta*D_j))
            # grad_log_softmax_k = exp(beta*D_k) / sum(exp(beta*D_j))
            
            # Let's compute weights
            # For softmin, weights are proportional to exp(-beta * D)
            # For softmax, weights are proportional to exp(beta * D)
            
            # Shift to prevent overflow
            # For softmin: subtract min(D) from D before exp? 
            # Let D' = D - min(D). Then exp(-beta*(D' + min(D))) = exp(-beta*min(D)) * exp(-beta*D')
            # The constant factor cancels in the normalization sum.
            # So we can just use exp(-beta * D) if D is small enough.
            # Max D is 2. Beta is 100. exp(-200) is ~1e-87. This is fine for double precision? 
            # exp(200) is ~1e86. This is fine.
            
            exp_neg_beta_D = np.exp(-beta * dists)
            exp_pos_beta_D = np.exp(beta * dists)
            
            sum_neg = np.sum(exp_neg_beta_D)
            sum_pos = np.sum(exp_pos_beta_D)
            
            # Weights for each pair k
            # w_min_k = exp(-beta * D_k) / sum_neg  (proportional to gradient of log softmin)
            # w_max_k = exp(beta * D_k) / sum_pos   (proportional to gradient of log softmax)
            
            w_min = exp_neg_beta_D / sum_neg
            w_max = exp_pos_beta_D / sum_pos
            
            # Gradient of log(softmin) w.r.t D_k is -w_min_k
            # Gradient of log(softmax) w.r.t D_k is w_max_k
            # Objective L = log(softmin) - log(softmax)
            # dL/dD_k = -w_min_k - w_max_k
            
            grad_D = -w_min - w_max
            
            # Now map grad_D to grad_P
            # D_k = ||p_i - p_j||
            # dD_k/dp_i = (p_i - p_j) / D_k
            # dD_k/dp_j = (p_j - p_i) / D_k
            
            grad_P = np.zeros((n, d))
            
            # Vectorized update
            # For each pair k with index (i, j) and weight w_k = grad_D[k]
            # grad_P[i] += w_k * (p_i - p_j) / D_k
            # grad_P[j] += w_k * (p_j - p_i) / D_k
            
            # We can do this in a loop for n=91 pairs, it's fast enough in Python if optimized?
            # Or use np.add.at
            # Let's use a loop for clarity and speed (91 iterations is trivial)
            
            for k in range(len(dists)):
                w = grad_D[k]
                if abs(w) < 1e-12:
                    continue
                i_idx, j_idx = pair_indices[k]
                diff = points[i_idx] - points[j_idx]
                dist_k = dists[k]
                if dist_k < 1e-10:
                    continue
                vec = diff / dist_k
                grad_P[i_idx] += w * vec
                grad_P[j_idx] -= w * vec
            
            # Update with momentum
            velocity = mu * velocity + alpha * grad_P
            points += velocity
            
            # Project back to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            points /= norms
            
            # Decay step size
            alpha *= 0.999
            
        # Final check after loop
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max > 0:
            final_score = d_min / d_max
            if final_score > best_score:
                best_score = final_score
                best_points = points.copy()

    if best_points is None:
        # Fallback to random if something went wrong
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
