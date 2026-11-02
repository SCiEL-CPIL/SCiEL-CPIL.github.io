"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _max_overlap(h):
    """Compute the maximum overlap R_k for k=1..N-1."""
    corrs = _compute_overlaps(h)
    return np.max(corrs)

def _binary_local_search(h_init, time_budget=10.0):
    """
    Perform simulated annealing on the binary sequence h.
    h must be binary (0 or 1) and sum to N/2.
    Moves are swaps of a 1 and a 0.
    """
    n = len(h)
    h = h_init.copy()
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    current_score = _max_overlap(h)
    best_score = current_score
    best_h = h.copy()
    
    T = 1.0
    T_min = 1e-4
    alpha = 0.99
    
    start_time = time.time()
    while _time_remaining() > time_budget and (time.time() - start_time) < time_budget:
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        # Pick random i from ones, j from zeros
        i_idx = np.random.choice(len(ones))
        j_idx = np.random.choice(len(zeros))
        i = ones[i_idx]
        j = zeros[j_idx]
        
        # Perform swap
        h[i] = 0
        h[j] = 1
        
        # Update lists (simplified: just find indices, N is small)
        # For efficiency, we can just recompute max_overlap. 
        # N=100, this is fast.
        new_score = _max_overlap(h)
        
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-6)):
            # Accept
            current_score = new_score
            # Update lists
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Reject
            h[i] = 1
            h[j] = 0
            
        T *= alpha
        
    return best_h, best_score

def _continuous_refine(h_init, time_budget=10.0):
    """
    Refine the binary solution using continuous optimization.
    Objective: minimize soft_max(R_k).
    Constraints: sum(h) = N/2, 0 <= h <= 1.
    """
    n = len(h_init)
    target_sum = n / 2.0
    
    def soft_max_obj_and_grad(h, beta=50.0):
        corrs = _compute_overlaps(h)
        # Soft max
        max_val = np.max(corrs)
        # To avoid overflow, subtract max
        shifted = corrs - max_val
        log_sum = np.log(np.sum(np.exp(beta * shifted)))
        soft_max = max_val + log_sum / beta
        
        # Gradient of soft max
        # d(soft_max)/d(corrs_j) = exp(beta * corrs_j) / sum(exp(beta * corrs))
        exp_beta = np.exp(beta * corrs)
        sum_exp = np.sum(exp_beta)
        weights = exp_beta / sum_exp
        
        # d(soft_max)/d(h) = sum_j weights_j * d(corrs_j)/d(h)
        grad_h = np.zeros(n)
        for k in range(1, n):
            w = weights[k-1] * (2.0 / n)
            if w == 0: continue
            grad_h[:n-k] += w * h[k:]
            grad_h[k:] += w * h[:n-k]
            
        return soft_max, grad_h

    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - target_sum, 'jac': lambda h: np.ones(n)}
    ]
    bounds = [(0, 1)] * n
    
    start_time = time.time()
    best_h = h_init.copy()
    best_score = _max_overlap(h_init)
    
    # Try a few betas for soft max
    for beta in [10.0, 50.0, 100.0]:
        if _time_remaining() < time_budget + 5:
            break
        try:
            res = minimize(
                soft_max_obj_and_grad,
                best_h,
                args=(beta,),
                method='L-BFGS-B', # L-BFGS-B doesn't support equality constraints directly in standard scipy versions without workarounds, 
                # so we might need to use SLSQP or project. 
                # Let's use SLSQP for small N.
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
            )
            
            # Check result
            h_new = res.x
            # Ensure bounds and sum
            h_new = np.clip(h_new, 0, 1)
            diff = target_sum - np.sum(h_new)
            if abs(diff) > 1e-9:
                mask = (h_new > 0) & (h_new < 1)
                if np.sum(mask) > 0:
                    h_new[mask] += diff / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
            
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                
        except Exception:
            pass
            
    return best_h, best_score

def construct():
    n = N
    target_sum = n / 2.0
    
    # 1. Binary Search Phase
    # We will run several restarts of the binary local search
    # Each restart gets a portion of the time
    best_binary_h = None
    best_binary_score = float('inf')
    
    # Time allocation: 80% for binary search, 20% for continuous refinement
    binary_time_budget = 100.0
    refine_time_budget = 20.0
    
    np.random.seed(42)
    
    # Initial guesses
    # 1. Random
    # 2. Alternating (bad, but maybe good start for search?)
    # 3. Block
    # Let's just do random restarts.
    
    num_restarts = 5
    time_per_restart = binary_time_budget / num_restarts
    
    for r in range(num_restarts):
        if _time_remaining() < refine_time_budget + 5:
            break
            
        # Create random binary sequence with 50 ones
        h_init = np.zeros(n)
        ones_indices = np.random.choice(n, 50, replace=False)
        h_init[ones_indices] = 1.0
        
        # Run local search
        h_res, score_res = _binary_local_search(h_init, time_budget=time_per_restart)
        
        if score_res < best_binary_score:
            best_binary_score = score_res
            best_binary_h = h_res
            
    # 2. Continuous Refinement Phase
    if best_binary_h is not None and _time_remaining() > refine_time_budget:
        h_final, score_final = _continuous_refine(best_binary_h, time_budget=refine_time_budget)
        if score_final < best_binary_score:
            best_binary_h = h_final
            best_binary_score = score_final
    else:
        h_final = best_binary_h if best_binary_h is not None else np.full(n, 0.5)
        
    # Final check and adjustment
    h_final = np.clip(h_final, 0, 1)
    diff = target_sum - np.sum(h_final)
    if abs(diff) > 1e-9:
        mask = (h_final > 0) & (h_final < 1)
        if np.sum(mask) > 0:
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0, 1)
            
    return h_final
# EVOLVE-BLOCK-END
