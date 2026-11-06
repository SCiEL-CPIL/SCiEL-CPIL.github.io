import numpy as np
from scipy.optimize import minimize

def construct():
    N = 300
    rng = np.random.default_rng(42)
    
    # Initialization: Fibonacci disk
    indices = np.arange(N)
    r = np.sqrt((indices + 0.5) / N) * 0.95
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack([x, y])
    
    # Ensure inside
    radii = np.linalg.norm(centers, axis=1)
    mask = radii >= 1.0
    if np.any(mask):
        centers[mask] = centers[mask] / radii[mask, np.newaxis] * 0.99

    # Precompute indices for pairs
    triu_i, triu_j = np.triu_indices(N, k=1)
    
    def objective_and_grad(flat_centers):
        c = flat_centers.reshape(N, 2)
        
        # Pairwise distances
        diff = c[triu_i] - c[triu_j]
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)
        pair_vals = dist / 2.0
        
        # Boundary distances
        radii = np.linalg.norm(c, axis=1)
        boundary_vals = 1.0 - radii
        
        # Combine values
        # We want to minimize -soft_min, which is equivalent to minimizing log-sum-exp(-alpha * vals)
        # Let's just compute the soft min directly and negate it for minimization?
        # No, let's stick to H = 1/alpha * log(sum(exp(-alpha * v)))
        # We want to MINIMIZE H.
        
        alpha = 50.0 # Tuning parameter
        
        # Calculate weights w_k = exp(-alpha * v_k) / sum(exp(-alpha * v_k))
        # To avoid underflow/overflow, shift by max
        all_vals = np.concatenate([pair_vals, boundary_vals])
        
        # Shift
        max_val = np.max(all_vals)
        exp_terms = np.exp(-alpha * (all_vals - max_val))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # H = max_val - (1/alpha) * log(sum_exp)
        # Wait, 
        # S = sum exp(-alpha v_i) = exp(-alpha max_val) sum exp(-alpha(v_i - max_val))
        # log S = -alpha max_val + log(sum_exp_shifted)
        # H = 1/alpha log S = -max_val + 1/alpha log(sum_exp_shifted)
        # This H is close to -max_val? No.
        # If v_i are all equal to v, S = N exp(-alpha v). log S = log N - alpha v.
        # H = 1/alpha (log N - alpha v) = v + (log N)/alpha.
        # So H approximates the mean of the smallest values?
        # Actually, as alpha -> inf, H -> min(v).
        # Let's check the sign.
        # We want to maximize min(v).
        # So we want to minimize -min(v).
        # Let F = -H. Then F approximates -min(v).
        # Minimizing F is maximizing min(v).
        
        # So objective = -H = max_val - 1/alpha * log(sum_exp)
        
        obj_val = max_val - (1.0 / alpha) * np.log(sum_exp)
        
        # Gradient
        # grad H = - sum w_k grad v_k
        # grad F = - grad H = sum w_k grad v_k
        
        # We need grad v_k for each k.
        # For pairs: v_k = d_ij / 2.
        # grad_{c_i} v_k = (c_i - c_j) / (2 * d_ij)
        # grad_{c_j} v_k = (c_j - c_i) / (2 * d_ij)
        
        # For boundary: v_k = 1 - |c_i|.
        # grad_{c_i} v_k = -c_i / |c_i|
        
        grad_c = np.zeros((N, 2))
        
        # Pair gradients
        # weights_pair = weights[:len(pair_vals)]
        # diff = c[triu_i] - c[triu_j]
        # dist = pair_vals * 2
        # unit_vec = diff / dist[:, None]
        # grad_contrib_i = unit_vec * weights_pair[:, None] / 2.0  # d/dci (d/2) = (ci-cj)/d * 1/2
        # grad_contrib_j = -unit_vec * weights_pair[:, None] / 2.0
        
        # Vectorized accumulation
        # This is tricky to do in one shot without loops or advanced indexing accumulation.
        # We can use np.add.at
        
        w_pairs = weights[:len(pair_vals)]
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-6, dist, 1.0)
        unit_vec = diff / dist_safe[:, None]
        
        # grad_i += unit_vec * w / 2
        # grad_j -= unit_vec * w / 2
        
        np.add.at(grad_c, triu_i, unit_vec * w_pairs[:, None] * 0.5)
        np.add.at(grad_c, triu_j, -unit_vec * w_pairs[:, None] * 0.5)
        
        # Boundary gradients
        w_bounds = weights[len(pair_vals):]
        radii_safe = np.where(radii > 1e-6, radii, 1.0)
        bound_grads = -c / radii_safe[:, None]
        grad_c += bound_grads * w_bounds[:, None]
        
        # The gradient of the objective F is grad_c
        # We are minimizing F, so the gradient returned to scipy should be grad_c
        # Because scipy expects grad of the function being minimized.
        
        return obj_val, grad_c

    # Use minimize
    # We need to flatten centers
    x0 = centers.flatten()
    
    # We might need multiple restarts or a high alpha
    # Let's try one run with a good alpha
    
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': 2000, 'gtol': 1e-8, 'disp': False})
    
    best_centers = res.x.reshape(N, 2)
    
    # Post-process to ensure strict inside
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.999
        
    return best_centers
