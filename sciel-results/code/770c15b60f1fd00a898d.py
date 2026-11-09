# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_P(support, weights):
    """
    Computes P[X1 + X2 + X3 < 2*X4] for discrete measure defined by support and weights.
    """
    n = len(support)
    # Ensure float64
    support = np.asarray(support, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights just in case, though optimizer should keep sum=1
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w_norm = weights / w_sum
    
    P = 0.0
    # Precompute pairwise sums and their probabilities?
    # s_i + s_j < threshold
    # We can precompute the CDF of the sum of two variables?
    # For small n, direct loop is fine.
    
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Calculate Prob[X1 + X2 < threshold]
            prob_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if support[i] + support[j] < threshold:
                        prob_sum += w_norm[i] * w_norm[j]
            P += w_norm[k] * w_norm[l] * prob_sum
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_P = 0.0
    best_support = None
    best_weights = None
    
    # Candidate support structures
    # The idea is to have some small values to keep X1+X2+X3 small,
    # and some large values to make 2*X4 large.
    candidates = [
        [0.0, 1.0, 10.0],
        [0.0, 1.0, 2.0, 10.0],
        [0.0, 0.5, 1.0, 10.0],
        [0.0, 1.0, 3.0, 10.0],
        [0.0, 1.0, 10.0, 20.0],
        [0.0, 1.0, 2.0, 3.0, 10.0]
    ]
    
    for support in candidates:
        if time.time() > deadline:
            break
            
        support_arr = np.array(support, dtype=np.float64)
        n = len(support_arr)
        
        # Parameterize weights using softmax to ensure positivity and sum=1
        # v in R^n -> w = exp(v) / sum(exp(v))
        
        def obj(v):
            # Softmax with numerical stability
            v_max = np.max(v)
            exp_v = np.exp(v - v_max)
            w = exp_v / np.sum(exp_v)
            return -_compute_P(support_arr, w)
        
        # Initial guess: uniform
        v0 = np.zeros(n)
        
        try:
            # Nelder-Mead is robust for small n
            res = minimize(obj, v0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.success:
                v_opt = res.x
                # Compute final weights
                v_max = np.max(v_opt)
                exp_v = np.exp(v_opt - v_max)
                w_opt = exp_v / np.sum(exp_v)
                
                P_opt = -res.fun
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = support_arr.copy()
                    best_weights = w_opt.copy()
                    
        except Exception:
            pass

    # Fallback to seed program result if optimization fails or finds nothing
    if best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure valid output format
        best_weights = np.clip(best_weights, 0.0, None)
        sum_w = np.sum(best_weights)
        if sum_w > 1e-9:
            best_weights = best_weights / sum_w
        else:
            # Invalid, fallback
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights


# EVOLVE-BLOCK-END
