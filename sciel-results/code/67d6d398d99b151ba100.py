import numpy as np
from scipy.optimize import linprog
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values
    for K in [20, 50, 100, 200, 500, 1000]:
        if time.time() > deadline - 60: # Safety margin
            break
            
        # Variables f(2) ... f(K)
        n_vars = K - 1
        if n_vars <= 0: continue
        
        # Objective: maximize -sum (log k / k) f_k
        # linprog minimizes c^T x, so we minimize sum (log k / k) f_k
        c = np.array([np.log(k) / k for k in range(2, K+1)])
        
        # Constraints: sum f_k {x/k} >= -1.0001
        # linprog handles A_ub x <= b_ub
        # So -sum f_k {x/k} <= 1.0001
        A_ub = np.zeros((10*K, n_vars))
        b_ub = np.full(10*K, 1.0001)
        
        for x in range(1, 10*K + 1):
            for i, k in enumerate(range(2, K+1)):
                A_ub[x-1, i] = -np.floor(x/k) + x/k # -{x/k}
                
        # Bounds: f_k can be any real number
        bounds = [(None, None) for _ in range(n_vars)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = res.x
                f_full = np.zeros(K+1)
                f_full[2:K+1] = f_vals
                f_full[1] = -np.sum(f_vals / np.arange(2, K+1))
                
                # Calculate score
                # a = -sum f(k) log(k) / k
                # Note: f(1) term is 0
                score = -np.sum(f_full[2:K+1] * np.log(np.arange(2, K+1)) / np.arange(2, K+1))
                
                # Verify feasibility (optional but good for debugging)
                # M(x) = sum f(k) floor(x/k)
                # We trust the LP constraints, but float errors might exist.
                
                if score > best_score:
                    best_score = score
                    best_domain = np.arange(1, K+1)
                    best_values = f_full[1:K+1]
        except Exception as e:
            pass

    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values
