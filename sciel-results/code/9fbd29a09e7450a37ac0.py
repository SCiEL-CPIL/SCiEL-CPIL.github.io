# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a smooth approximation of the min/max distance ratio via softmax/softmin
    and multi-start L-BFGS-B optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute pair indices
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    n_pairs = len(pair_indices)
    i_idx, j_idx = pair_indices.T

    # Temperature for softmin/softmax
    # Smaller tau makes it closer to hard min/max but less smooth
    tau = 0.05 

    def objective_and_grad(x):
        points = x.reshape(n, d)
        
        # Compute all pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        
        # Extract pairwise squared distances (upper triangle)
        pair_sq_dists = sq_dists[i_idx, j_idx]
        
        # Softmax of sq_dists
        # max_val for numerical stability
        max_sq = np.max(pair_sq_dists)
        exp_sq = np.exp((pair_sq_dists - max_sq) / tau)
        softmax_sq = exp_sq / np.sum(exp_sq)
        
        # Softmin of sq_dists is -Softmax(-sq_dists)
        min_sq = -np.min(pair_sq_dists)
        exp_neg_sq = np.exp((-pair_sq_dists - min_sq) / tau)
        softmax_neg_sq = exp_neg_sq / np.sum(exp_neg_sq)
        
        # log(softmax) = max_sq/tau + log(sum(exp))
        log_softmax_sq = max_sq / tau + np.log(np.sum(exp_sq))
        
        # log(softmin) = min_sq/tau + log(sum(exp_neg_sq))  -- Wait, let's derive carefully
        # SoftMin(x, tau) = -tau * log(sum(exp(-x_i/tau)))
        # log(SoftMin) = log(-tau) - log(sum(exp(-x_i/tau)))
        # This is getting complicated with signs. Let's use the gradient form directly.
        
        # We want to maximize log(SoftMin) - log(SoftMax)
        # SoftMin(x, tau) = -tau * log(sum(exp(-x_i/tau)))
        # log(SoftMin) = log(-tau) - log(sum(exp(-x_i/tau)))
        # d/dx log(SoftMin) = - (1/sum(exp(-x/tau))) * sum(exp(-x/tau) * (-1/tau) * d(x_i)/dx)
        # Let P_min_i = exp(-x_i/tau) / sum(exp(-x_k/tau)). This is the "min-softmax" weights.
        # d/dx log(SoftMin) = (1/tau) * sum(P_min_i * d(x_i)/dx)
        
        # SoftMax(x, tau) = tau * log(sum(exp(x_i/tau)))
        # log(SoftMax) = log(tau) + log(sum(exp(x_i/tau)))
        # Let P_max_i = exp(x_i/tau) / sum(exp(x_k/tau)). This is the standard softmax weights.
        # d/dx log(SoftMax) = (1/tau) * sum(P_max_i * d(x_i)/dx)
        
        # Objective to minimize: - (log(SoftMin) - log(SoftMax))
        # Gradient: - ( (1/tau) * sum(P_min_i * d(x_i)/dx) - (1/tau) * sum(P_max_i * d(x_i)/dx) )
        #           = (1/tau) * sum( (P_max_i - P_min_i) * d(x_i)/dx )
        
        # Calculate weights
        # P_max
        exp_sq = np.exp((pair_sq_dists - max_sq) / tau)
        sum_exp_sq = np.sum(exp_sq)
        P_max = exp_sq / sum_exp_sq
        
        # P_min
        # x_i = pair_sq_dists
        # -x_i/tau
        min_val = np.min(pair_sq_dists)
        exp_neg_sq = np.exp((-pair_sq_dists - min_val) / tau)
        sum_exp_neg_sq = np.sum(exp_neg_sq)
        P_min = exp_neg_sq / sum_exp_neg_sq
        
        # Objective value (for monitoring, not strictly needed for L-BFGS-B if grad is good)
        # val = - (log(SoftMin) - log(SoftMax))
        # log(SoftMin) = log(-tau) - log(sum(exp_neg_sq)) + min_val/tau ? No.
        # SoftMin = -tau * log(sum(exp(-x/tau)))
        # log(SoftMin) = log(-tau) - log(sum(exp(-x/tau)))
        # log(SoftMax) = log(tau) + log(sum(exp(x/tau)))
        # Ratio term = log(SoftMin) - log(SoftMax) = -2log(tau) - log(sum(exp(-x/tau))) - log(sum(exp(x/tau)))
        # We minimize -RatioTerm
        
        # Gradient calculation
        # d(x_i)/dx_k for x_i = ||p_a - p_b||^2
        # d(x_i)/dp_a = 2(p_a - p_b)
        # d(x_i)/dp_b = 2(p_b - p_a)
        
        # coeff_i = (P_max_i - P_min_i) / tau
        coeffs = (P_max - P_min) / tau
        
        grad = np.zeros_like(x)
        for k in range(n_pairs):
            c = coeffs[k]
            if abs(c) < 1e-12:
                continue
            i, j = pair_indices[k]
            diff_vec = points[i] - points[j]
            # grad[p_i] += c * 2 * diff_vec
            # grad[p_j] += c * 2 * (-diff_vec)
            grad[i*d:(i+1)*d] += 2.0 * c * diff_vec
            grad[j*d:(j+1)*d] -= 2.0 * c * diff_vec
        
        # Objective value for the optimizer (it needs the value)
        # We need a scalar value.
        # val = - (log(SoftMin) - log(SoftMax))
        # log(SoftMin) = log(-tau) - log(sum_exp_neg_sq)  (ignoring the shift for constant term, but let's be precise)
        # Actually, let's just compute the soft min and max values.
        soft_min_val = -tau * np.log(sum_exp_neg_sq) + min_val  # Wait, exp(-x/tau) = exp((-x - min_val)/tau) * exp(-min_val/tau)
        # sum(exp(-x/tau)) = exp(-min_val/tau) * sum(exp((-x-min_val)/tau))
        # log(sum(exp(-x/tau))) = -min_val/tau + log(sum_exp_neg_sq)
        # SoftMin = -tau * (-min_val/tau + log(sum_exp_neg_sq)) = min_val - tau * log(sum_exp_neg_sq)
        
        soft_max_val = tau * np.log(sum_exp_sq) + max_sq
        
        obj = -(np.log(soft_min_val) - np.log(soft_max_val))
        
        return obj, grad

    # Initialization strategies
    def generate_initial_points(seed):
        rng = np.random.RandomState(seed)
        # Mix of spherical and random
        if seed % 2 == 0:
            # Fibonacci sphere
            points = np.zeros((n, 3))
            golden_angle = np.pi * (3. - np.sqrt(5.))
            for i in range(n):
                y = 1 - (i / (n - 1)) * 2
                radius = np.sqrt(1 - y**2)
                theta = golden_angle * i
                points[i, 0] = np.cos(theta) * radius
                points[i, 1] = np.sin(theta) * radius
                points[i, 2] = y
            # Add small noise
            points += rng.randn(n, 3) * 0.1
        else:
            # Random uniform in sphere
            points = rng.randn(n, 3)
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * rng.uniform(0.5, 1.0, n)[:, np.newaxis]
        
        return points.flatten()

    best_points = None
    best_score = -np.inf

    # Try multiple starts
    for seed in range(10):
        x0 = generate_initial_points(seed)
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x
            points = x_opt.reshape(n, 3)
            
            # Evaluate actual metric
            d = pdist(points)
            if d.max() == 0:
                score = 0
            else:
                score = (d.min() / d.max()) ** 2
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue
            
    # If all failed, fallback
    if best_points is None:
        best_points = generate_initial_points(0).reshape(n, 3)
        
    return best_points

# EVOLVE-BLOCK-END
