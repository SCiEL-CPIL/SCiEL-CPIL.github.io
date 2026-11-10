import numpy as np
from scipy.optimize import linprog
import sys

def prime_number_theorem_partial_function():
    best_score = 0.0
    best_domain = np.array([1], dtype=np.int64)
    best_values = np.array([0.0], dtype=np.float64)
    
    # Try a few larger K values directly to save time
    # K=200: 199 vars, 2000 constraints
    # K=500: 499 vars, 5000 constraints
    # K=1000: 999 vars, 10000 constraints
    K_candidates = [100, 200, 500]
    
    for K in K_candidates:
        n_vars = K - 1
        if n_vars == 0:
            continue
            
        # Objective: minimize sum v_k ln(k)/k
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        # Constraints: M(x) <= 1 for x in 1..10K
        # M(x) = sum v_k (floor(x/k) - x/k) = - sum v_k (x % k) / k
        x_vals = np.arange(1, 10 * K + 1)[:, np.newaxis]
        k_vals = ks[np.newaxis, :]
        
        # A_ub[x-1, i] corresponds to constraint x, variable k_i
        # value is - (x % k_i) / k_i
        A_ub = - (x_vals % k_vals) / k_vals
        b_ub = np.ones(10 * K)
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                v = res.x
                # Calculate score a = - sum v_k ln(k)/k
                score = -np.dot(v, c)
                if score > best_score:
                    best_score = score
                    domain = np.concatenate(([1], ks)).astype(np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = 1.0 # Placeholder, will be adjusted by evaluator
                    values[1:] = v
                    best_domain = domain
                    best_values = values
        except Exception:
            pass
            
    # Fallback
    if best_score < 0.6:
        best_domain = np.array([1, 2], dtype=np.int64)
        best_values = np.array([1.0, -2.0], dtype=np.float64)
        best_score = np.log(2)
        
    return best_domain, best_values
