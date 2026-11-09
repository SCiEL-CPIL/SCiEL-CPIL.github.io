import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Parameterize the 10x10 probability matrix P using a rank-1 model:
       P[y, x] = exp(a_y + b_x) / Z.
       This reduces the 100-dimensional simplex search to a 20-parameter 
       optimization problem (10 row weights, 10 column weights).
       This structure captures the "product" nature often found in optimal 
       Kakeya-type distributions and drastically reduces the risk of 
       getting stuck in local minima compared to full-dimension search.
    2. Use a deterministic multi-start approach with L-BFGS-B on the 
       log-parameters (a, b). L-BFGS-B is highly efficient for this 
       smooth, low-dimensional objective.
    3. Return the best valid result found within the time budget.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 590  # 10s buffer for overhead

    # Pre-compute indices for entropy calculation
    # We need to map each cell (y, x) to its bin indices for the 4 distributions
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9 -> 0..18
    idx_x   = xs.astype(np.int32)                      # x -> 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y -> 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y -> 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
        """Calculate Shannon entropy (bits) from a count/probability vector."""
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        p = counts[mask]
        p_sum = p.sum()
        if p_sum < 1e-15:
            return 0.0
        p = p / p_sum
        return float(-np.sum(p * np.log2(p)))

    def evaluate_P(P_flat):
        """
        Evaluate the score for a flattened probability vector P_flat.
        """
        if not np.all(np.isfinite(P_flat)):
            return -np.inf
        
        # Clip and normalize
        P_norm = np.clip(P_flat, 0, None)
        s = P_norm.sum()
        if s < 1e-9:
            return -np.inf
        P_norm /= s

        c_dm1 = np.bincount(idx_dm1, weights=P_norm, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P_norm, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P_norm, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P_norm, minlength=bin_sizes['d2'])

        H_dm1 = entropy_from_counts(c_dm1)
        H_x   = entropy_from_counts(c_x)
        H_d1  = entropy_from_counts(c_d1)
        H_d2  = entropy_from_counts(c_d2)

        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        if M < 1e-9:
            return -np.inf
            
        return D / M

    def score_from_params(params):
        """
        Objective function for minimization.
        Input: params = [a_0..a_9, b_0..b_9]
        Output: Negative score
        """
        a = params[:n]
        b = params[n:]
        
        # Compute unnormalized log-probs
        # P[y, x] proportional to exp(a_y + b_x)
        # log(P[y, x]) = a_y + b_x - log(Z)
        # Z = sum_{y,x} exp(a_y + b_x) = (sum_y exp(a_y)) * (sum_x exp(b_x))
        
        # To avoid overflow/underflow, we can work with shifted values or just let exp handle it 
        # if values are reasonable. L-BFGS-B keeps them in bounds.
        
        # However, for numerical stability in the objective function itself, 
        # we can compute the normalized probabilities directly.
        
        # log_Z = log(sum exp(a)) + log(sum exp(b))
        # max_a = max(a), max_b = max(b)
        # sum exp(a) = exp(max_a) * sum exp(a - max_a)
        
        max_a = np.max(a)
        max_b = np.max(b)
        
        sum_exp_a = np.sum(np.exp(a - max_a))
        sum_exp_b = np.sum(np.exp(b - max_b))
        
        if sum_exp_a < 1e-15 or sum_exp_b < 1e-15:
            return 1e9
            
        # log_Z = max_a + log(sum_exp_a) + max_b + log(sum_exp_b)
        log_Z = max_a + np.log(sum_exp_a) + max_b + np.log(sum_exp_b)
        
        # log_p[y, x] = a_y + b_x - log_Z
        # We can construct the 10x10 matrix of log probabilities
        log_p_matrix = a[:, None] + b[None, :] - log_Z
        p_matrix = np.exp(log_p_matrix)
        
        # Flatten for evaluation
        p_flat = p_matrix.flatten()
        
        score = evaluate_P(p_flat)
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Multi-Start Optimization ---
    best_score = -np.inf
    best_params = None
    
    # Define several diverse starting points for (a, b)
    # 1. Uniform (a=0, b=0)
    # 2. Diagonal bias (a_i = i, b_i = -i)
    # 3. Anti-diagonal bias (a_i = i, b_i = i)
    # 4. Random initializations
    
    rng = np.random.default_rng(42)
    
    starts = []
    # Uniform
    starts.append(np.zeros(2 * n))
    
    # Diagonal
    diag_a = np.linspace(-5, 5, n)
    diag_b = -diag_a
    starts.append(np.concatenate([diag_a, diag_b]))
    
    # Anti-diagonal
    anti_diag_a = np.linspace(-5, 5, n)
    anti_diag_b = anti_diag_a
    starts.append(np.concatenate([anti_diag_a, anti_diag_b]))
    
    # Some random starts
    for _ in range(5):
        rand_a = rng.uniform(-10, 10, n)
        rand_b = rng.uniform(-10, 10, n)
        starts.append(np.concatenate([rand_a, rand_b]))
    
    bounds = list(zip(np.full(2*n, -20), np.full(2*n, 20)))
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        try:
            result = minimize(
                score_from_params,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_params = result.x
                
        except Exception as e:
            pass

    # --- Final Conversion and Validation ---
    if best_params is None:
        # Fallback to uniform
        P_final = np.ones(N) / N
    else:
        a = best_params[:n]
        b = best_params[n:]
        
        # Reconstruct P
        max_a = np.max(a)
        max_b = np.max(b)
        sum_exp_a = np.sum(np.exp(a - max_a))
        sum_exp_b = np.sum(np.exp(b - max_b))
        log_Z = max_a + np.log(sum_exp_a) + max_b + np.log(sum_exp_b)
        
        log_p_matrix = a[:, None] + b[None, :] - log_Z
        P_final = np.exp(log_p_matrix).flatten()
        
        # Clip and normalize just in case of numerical drift
        P_final = np.clip(P_final, 0, None)
        s = P_final.sum()
        if s > 1e-9:
            P_final /= s
        else:
            P_final = np.ones(N) / N
            
    P_final = P_final.reshape((n, n))
    
    # Final safety check
    if not np.all(np.isfinite(P_final)) or P_final.shape != (n, n):
        return np.ones((n, n)) / N
        
    return P_final

if __name__ == "__main__":
    pass
