import numpy as np
import time
import math
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600  # 600s hard limit
    
    # Baseline from seed
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    def solve_lp(K, time_remaining):
        if time_remaining < 10:
            return None
            
        n_vars = K - 1
        if n_vars <= 0:
            return None
        
        # Objective: minimize sum w_k * log(k)/k
        c = np.array([math.log(k) / k for k in range(2, K + 1)])
        
        max_x = 10 * K
        n_constraints = max_x
        
        # Build A_ub
        # A_ub[i, j] = frac((i+1) / (j+2))
        # Using vectorized operations for speed
        xs = np.arange(1, max_x + 1, dtype=np.float64)
        ks = np.arange(2, K + 1, dtype=np.float64)
        
        # Reshape for broadcasting
        # xs shape (n_constraints, 1)
        # ks shape (1, n_vars)
        A_ub = (xs[:, np.newaxis] / ks[np.newaxis, :]) - np.floor(xs[:, np.newaxis] / ks[np.newaxis, :])
        
        b_ub = np.full(n_constraints, 1.0001)
        bounds = [(-10, 10)] * n_vars
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None
        
        if not result.success:
            return None
        
        w = result.x
        v = -w
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[0] = 0.0
        values[1:] = v
        
        # Verification
        values_clipped = np.clip(values, -10, 10)
        S = np.sum(values_clipped / domain.astype(float))
        f_adj = values_clipped.copy()
        f_adj[0] = f_adj[0] - S
        
        # Check M(x)
        # Efficient check:
        # M(x) = sum f_adj(k) * floor(x/k)
        # We can compute this for all x in [1, 10K]
        # Using convolution or direct sum?
        # Direct sum is O(K * 10K) = 10 K^2.
        # For K=1000, 10^7 ops. Fast in numpy.
        
        max_x_check = 10 * K
        xs_check = np.arange(1, max_x_check + 1, dtype=np.float64)
        # floor(xs_check / domain)
        # domain is 1..K
        # We need sum over k of f_adj(k) * floor(x/k)
        # Let's construct the matrix or use loops?
        # Vectorized:
        # floor_matrix = np.floor(xs_check[:, None] / domain[None, :])
        # M_vals = floor_matrix @ f_adj
        # This creates a (10K x K) matrix. 10000 x 1000 = 10M floats = 80MB. OK.
        
        try:
            floor_matrix = np.floor(xs_check[:, np.newaxis] / domain[np.newaxis, :])
            M_vals = floor_matrix @ f_adj
            if np.any(M_vals > 1.0001):
                return None
        except Exception:
            return None
        
        a = -np.sum(f_adj * np.log(domain.astype(float)) / domain.astype(float))
        
        return domain, values, a

    # Try increasing K
    K_values = [200, 400, 600, 800, 1000, 1500, 2000]
    
    for K in K_values:
        elapsed = time.time() - start_time
        time_remaining = deadline - elapsed - 30
        if time_remaining < 20:
            break
            
        try:
            res = solve_lp(K, time_remaining)
            if res is not None:
                domain, values, a = res
                if a > best_score:
                    best_score = a
                    best_domain = domain.copy()
                    best_values = values.copy()
                    print(f"K={K}: score={a:.6f}, time={time.time()-start_time:.1f}s")
        except Exception as e:
            print(f"K={K} failed: {e}")
            continue
            
    return best_domain, best_values
