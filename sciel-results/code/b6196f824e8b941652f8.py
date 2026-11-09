import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values, extended to 3000
    for K in [50, 100, 200, 500, 1000, 2000, 3000]:
        if time.time() - start_time > 500:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        # Vectorized construction of A_ub
        # A_ub[i, j] = - { (i+1) / (j+2) }
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        
        try:
            # Memory: max_x * n_vars
            # For K=3000, 30000 * 3000 = 90M floats ~ 720MB. Acceptable.
            A_ub = -(x_vals[:, None] % ks[None, :]) / ks[None, :]
            b_ub = np.ones(max_x)
            
            # Add constraints to ensure f(1) = -sum(g/k) is in [-10, 10]
            # sum(g/k) <= 10
            # sum(g/k) >= -10  => -sum(g/k) <= 10
            
            row_pos = 1.0 / ks
            row_neg = -1.0 / ks
            
            A_ub_full = np.vstack([A_ub, row_pos, row_neg])
            b_ub_full = np.concatenate([b_ub, [10.0, 10.0]])
            
            bounds = [(-10, 10)] * n_vars
            
            res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs')
            
            if res.success:
                g = res.x
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                # Verify f(1) is within bounds
                if abs(f[1]) <= 10.0 + 1e-9:
                    a = -np.sum(g * c)
                    if a > best_score:
                        best_score = a
                        domain = np.arange(1, K + 1)
                        values = np.clip(f[1:K+1], -10, 10)
                        best_domain = domain
                        best_values = values

        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
