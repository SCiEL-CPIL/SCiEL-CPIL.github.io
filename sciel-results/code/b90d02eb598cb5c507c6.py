import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    def get_radius(centres):
        # Compute min pairwise distance / 2
        # and min boundary distance
        if len(centres) > 1:
            diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        else:
            min_pair = np.inf
        
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair, min_boundary)

    def objective_and_grad(x, beta):
        # x is flattened array of shape (N*3,)
        centres = x.reshape(-1, 3)
        
        # Pairwise distances
        # To avoid calculating full N x N matrix for gradient if N is large, 
        # but N=50 is small, so full matrix is fine.
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        # b_ik = 0.5 - |c_ik|
        # Shape (N, 3)
        b_dists = 0.5 - np.abs(centres)
        
        # Combine all distances
        # We need to sum exp(-beta * d) for all d
        # And sum exp(-beta * d) * grad_d for gradient
        
        # To be efficient and stable, we can compute the weights first
        # Weights w_d = exp(-beta * d) / sum(exp(-beta * d))
        
        # Flatten distances for easier handling
        # Pairwise: only upper triangle to avoid double counting? 
        # The formula sums over all i<j. 
        # If we use full matrix, we sum i!=j. 
        # The soft min is invariant to scaling of the sum, so double counting is fine as long as consistent.
        # However, for gradient, we need to be careful.
        
        # Let's stick to the full matrix but ignore diagonal.
        # dists is (N, N)
        # b_dists is (N, 3)
        
        # To prevent overflow in exp, we can subtract max d? No, we want small d to have large weight.
        # exp(-beta * d). If d is large, exp is small. If d is small, exp is large.
        # If beta is large and d is small, exp can be large.
        # Standard trick: subtract max of (-beta * d) ? No, that's for max.
        # For sum of exp, we can subtract max of the exponents to prevent overflow.
        # exponents = -beta * d
        # max_exp = max(exponents)
        # weights = exp(exponents - max_exp)
        # sum_weights = sum(weights)
        
        # Let's collect all exponents
        # Pairwise exponents: -beta * dists (N x N)
        # Boundary exponents: -beta * b_dists (N x 3)
        
        p_exps = -beta * dists
        b_exps = -beta * b_dists
        
        # Mask diagonal for pairwise
        mask = ~np.eye(N, dtype=bool)
        p_exps_masked = p_exps[mask]
        
        # Combine
        all_exps = np.concatenate([p_exps_masked, b_exps.flatten()])
        
        # Normalize to prevent overflow
        max_exp = np.max(all_exps)
        if max_exp == -np.inf:
             # All distances are inf? Unlikely.
             return 0.0, np.zeros_like(x)
            
        weights = np.exp(all_exps - max_exp)
        sum_weights = np.sum(weights)
        
        # Objective: -soft_min = (1/beta) * log(sum_weights) + max_exp/beta
        # Actually, soft_min = -1/beta * log(sum exp(-beta d))
        # log(sum exp(-beta d)) = log(sum exp(exps - max_exp) * exp(max_exp))
        #                        = max_exp + log(sum exp(exps - max_exp))
        # soft_min = -1/beta * (max_exp + log(sum_weights))
        # We want to MINIMIZE -soft_max ? No, we want to MAXIMIZE soft_min.
        # So we MINIMIZE -soft_min.
        # f = -soft_min = 1/beta * (max_exp + log(sum_weights))
        
        val = (max_exp + np.log(sum_weights)) / beta
        
        # Gradient
        # d(soft_min)/dx = 1/beta * (sum w_d * d(d)/dx) / sum(w_d)
        # Wait, d/dx log(sum exp(-beta d)) = (1/sum) * sum exp(-beta d) * (-beta * d(d)/dx)
        # So d(soft_min)/dx = -1/beta * [ sum w_d * (-beta * d(d)/dx) ] = sum w_d * d(d)/dx
        # Wait, w_d here is normalized weight?
        # Let W_d = exp(-beta d).
        # d(soft_min)/dx = 1/beta * (sum W_d * d(-beta d)/dx) / sum W_d 
        #                = 1/beta * (sum W_d * (-beta * d(d)/dx)) / sum W_d
        #                = - sum W_d * d(d)/dx / sum W_d
        # Since we want to MINIMIZE -soft_min, the gradient of the objective is:
        # grad_obj = - grad(soft_min) = sum W_d * d(d)/dx / sum W_d
        
        # We have weights = W_d / sum_W (normalized)
        # So grad_obj = sum (weights_d * d(d)/dx)
        
        grad = np.zeros((N, 3))
        
        # Pairwise contribution
        # We need to map weights back to the matrix
        # p_weights is the first part of 'weights'
        num_pairs = N * (N - 1)
        p_weights = weights[:num_pairs]
        b_weights = weights[num_pairs:]
        
        # Reconstruct p_weights matrix (N x N)
        p_w_mat = np.zeros((N, N))
        p_w_mat[mask] = p_weights
        
        # Gradient of distance d_ij w.r.t c_i is (c_i - c_j) / d_ij
        # Gradient of distance d_ij w.r.t c_j is (c_j - c_i) / d_ij
        # Sum over j for each i:
        # grad_i = sum_j w_ij * (c_i - c_j) / d_ij
        
        # Vectorized:
        # diff is (N, N, 3)
        # dists is (N, N)
        # We need to divide diff by dists
        # Handle dists=0 (diagonal) by masking
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_dirs = diff / dists[:, :, np.newaxis]
        
        # grad_dirs has inf on diagonal. Mask it.
        grad_dirs[~mask] = 0.0
        
        # grad_pair = sum_j w_ij * grad_dir_ij
        # p_w_mat is (N, N). grad_dirs is (N, N, 3).
        grad_pair = np.einsum('ij,ijk->ik', p_w_mat, grad_dirs)
        
        # Boundary contribution
        # b_dists is (N, 3)
        # d(b_ik)/d(c_ik) = -sign(c_ik)
        # grad_b = sum_k w_b_ik * (-sign(c_ik))
        # b_weights is flattened (N*3,)
        b_w_mat = b_weights.reshape(N, 3)
        
        # sign of centres
        signs = np.sign(centres)
        # If centre is 0, sign is 0. Gradient is undefined, but we can assume 0 or 1.
        # Let's use 0 for safety, or 1. The distance to boundary is 0.5 - |c|.
        # d/dc (0.5 - |c|) = -sign(c).
        grad_bound = -b_w_mat * signs
        
        grad = grad_pair + grad_bound
        
        # Flatten gradient
        grad_flat = grad.flatten()
        
        return val, grad_flat

    best_radius = 0.0
    best_centres = None
    
    # Restart loop
    np.random.seed(42)
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        # Jittered grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        indices = np.random.permutation(len(grid))[:N]
        centres = grid[indices].copy()
        
        # Add random jitter
        centres += np.random.normal(0, 0.02, centres.shape)
        centres = np.clip(centres, -0.45, 0.45)
        
        x0 = centres.flatten()
        
        # Optimization
        beta = 10.0
        best_x = x0.copy()
        best_val = np.inf
        
        for stage in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Define objective for this beta
            def obj_wrapper(x):
                val, grad = objective_and_grad(x, beta)
                return val, grad
            
            # Use L-BFGS-B
            # Bounds: [-0.5, 0.5] for each coordinate
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                res = minimize(
                    obj_wrapper,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x0 = res.x
            except Exception:
                # If optimization fails, break
                break
            
            # Update beta
            beta *= 2.0
            
            # Check time
            if time.time() - start_time > time_limit:
                break
        
        # Evaluate final radius
        final_centres = x0.reshape(-1, 3)
        current_radius = get_radius(final_centres)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = final_centres.copy()
            
        # Update x0 for next restart? No, we restart from scratch.
        # But we can use the best found so far as a seed for the next?
        # Let's just do independent restarts.
        
    if best_centres is None:
        # Fallback to simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
        
    return best_centres
