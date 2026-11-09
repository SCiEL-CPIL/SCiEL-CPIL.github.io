import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Strategy: 
    1. Multi-start local gradient optimization (L-BFGS-B) on log-probabilities.
    2. Diverse structured initializations to escape local optima.
    3. Time-budgeted search.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 620  # 10s buffer before 630s limit
    
    # Pre-compute indices for entropy calculation
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
        Assumes P_flat is non-negative and sums to ~1.
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

    def score_from_log_probs(log_p):
        """
        Objective function for optimization.
        Input: log_p (unnormalized log-probabilities)
        Output: Negative score (for minimization)
        """
        # Softmax to get probabilities
        # Subtract max for numerical stability
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9 # Large penalty
        p = p / s
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # Generate diverse initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros(N))
    
    # 2. Diagonal (x == y)
    diag = np.zeros(N)
    for i in range(n):
        diag[i * n + i] = 1.0
    initializations.append(np.log(diag + 1e-10))
    
    # 3. Anti-diagonal (x + y == 9)
    anti_diag = np.zeros(N)
    for i in range(n):
        anti_diag[i * n + (n - 1 - i)] = 1.0
    initializations.append(np.log(anti_diag + 1e-10))
    
    # 4. Corner (0,0)
    corner = np.zeros(N)
    corner[0] = 1.0
    initializations.append(np.log(corner + 1e-10))
    
    # 5. Random sparse
    for _ in range(5):
        mask = np.random.random(N) > 0.9
        if not np.any(mask):
            mask[0] = True
        rand_p = np.where(mask, 1.0, 0.0)
        rand_p /= rand_p.sum()
        initializations.append(np.log(rand_p + 1e-10))
        
    # 6. Random dense
    for _ in range(5):
        rand_p = np.random.random(N)
        rand_p /= rand_p.sum()
        initializations.append(np.log(rand_p + 1e-10))

    best_log_p = initializations[0]
    best_score = -1e9
    best_fun = 1e9

    # Multi-start optimization
    for i, init in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        # Check time per iteration
        if time.time() > deadline - 30:
            break

        try:
            result = minimize(
                score_from_log_probs, 
                init, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_log_p = result.x
                best_score = -result.fun
                
        except Exception:
            continue

    # Final Conversion and Validation
    log_p_shifted = best_log_p - np.max(best_log_p)
    P_final = np.exp(log_p_shifted)
    P_final = np.clip(P_final, 0, None)
    s = P_final.sum()
    if s > 1e-9:
        P_final /= s
    else:
        # Fallback to uniform if something went wrong
        P_final = np.ones(N) / N
        
    P_final = P_final.reshape((n, n))
    
    # Final safety check
    if not np.all(np.isfinite(P_final)) or P_final.shape != (n, n):
        return np.ones((n, n)) / N
        
    return P_final

if __name__ == "__main__":
    pass
