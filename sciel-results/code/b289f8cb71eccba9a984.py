import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Use Differential Evolution (DE) on log-probabilities for robust global search.
       DE is well-suited for non-convex problems like entropy ratios where local
       methods (L-BFGS-B) often get trapped in local optima.
    2. Refine the best solution from DE using L-BFGS-B for high-precision local convergence.
    3. Return the best valid result found within the time budget.
    """
    n = 10
    N = n * n
    start_time = time.time()
    # Keep a safety margin for final validation and return
    deadline = start_time + 590  # 10s buffer for overhead

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
        Objective function for minimization.
        Input: log_p (unnormalized log-probabilities)
        Output: Negative score
        """
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9 
        p = p / s
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Phase 1: Differential Evolution ---
    # Bounds for log-probs to prevent overflow/underflow issues in exp
    # We want a wide range to allow diverse structures
    lb = np.full(N, -20)
    ub = np.full(N, 20)
    
    # DE options
    # maxiter: limited to allow time for refinement
    # popsize: larger population for better exploration
    # tol: precision for stopping
    # update_interval: for potential monitoring, but we just want the result
    de_options = {
        'maxiter': 100,
        'popsize': 15, # 15 * 100 = 1500 evaluations. Each eval is fast (~1ms). 1.5s total.
        'tol': 1e-8,
        'mutation': (0.5, 1.5),
        'recombination': 0.7,
        'seed': 42
    }
    
    try:
        # DE minimizes the objective. Our objective is -score.
        # We want to maximize score, so minimizing -score is correct.
        result_de = differential_evolution(
            score_from_log_probs,
            bounds=list(zip(lb, ub)),
            options=de_options
        )
        
        best_log_p = result_de.x
        best_score = -result_de.fun
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to a good structured init if DE fails
        best_log_p = np.zeros(N)
        for i in range(n):
            best_log_p[i*n + i] = 10.0 # Diagonal
        best_score = -score_from_log_probs(best_log_p)

    # Check time
    if time.time() > deadline - 10:
        P_final = np.exp(best_log_p - np.max(best_log_p))
        P_final = np.clip(P_final, 0, None)
        s = P_final.sum()
        if s > 1e-9:
            P_final /= s
        else:
            P_final = np.ones(N) / N
        return P_final.reshape((n, n))

    # --- Phase 2: L-BFGS-B Refinement ---
    # Refine the best solution from DE
    lb_ref = np.full(N, -30)
    ub_ref = np.full(N, 30)
    
    l_bfgs_options = {
        'maxiter': 200,
        'ftol': 1e-12,
        'gtol': 1e-12
    }
    
    try:
        result_l_bfgs = minimize(
            score_from_log_probs,
            best_log_p,
            method='L-BFGS-B',
            bounds=list(zip(lb_ref, ub_ref)),
            options=l_bfgs_options
        )
        
        if result_l_bfgs.fun < -best_score:
            best_score = -result_l_bfgs.fun
            best_log_p = result_l_bfgs.x
            
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
        # Keep DE result

    # --- Final Conversion and Validation ---
    if not np.isfinite(best_score) or best_score < 0:
        # Fallback to uniform if optimization failed completely
        P_final = np.ones(N) / N
    else:
        log_p_shifted = best_log_p - np.max(best_log_p)
        P_final = np.exp(log_p_shifted)
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
