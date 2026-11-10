import numpy as np
from scipy.optimize import linprog
import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 630  # hard limit from evaluator
    safety_margin = 30  # seconds to spare
    
    best_result = None
    best_score = -np.inf
    
    # Try increasing K values. The LP has K-1 variables and 10K constraints.
    # For K=100: 99 vars, 1000 constraints -> fast
    # For K=200: 199 vars, 2000 constraints -> moderate
    # For K=400: 399 vars, 4000 constraints -> slower
    # For K=800: 799 vars, 8000 constraints -> slow
    # For K=1500: 1499 vars, 15000 constraints -> very slow
    
    # We'll try a sequence of K values, stopping if we run out of time
    K_values = [100, 200, 400, 800, 1500]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            break
        
        n_vars = K - 1
        
        # Objective: maximize - sum f(k) log(k)/k for k=2..K
        # linprog minimizes c^T x, so c = log(k)/k
        k_vals = np.arange(2, K + 1, dtype=np.float64)
        c = np.log(k_vals) / k_vals
        
        # Constraints: M(x) <= 1.0001 for x in [1, 10K]
        # After evaluator adjustment:
        # f_adj(1) = f(1) - S, where S = sum_{k in domain} f(k)/k
        # If we set f(1) = 0 (or don't include 1, effectively f(1)=0 before adjustment),
        # then S = sum_{k=2}^K f(k)/k
        # f_adj(1) = -S = -sum_{k=2}^K f(k)/k
        # 
        # M(x) = f_adj(1)*x + sum_{k=2}^K f(k)*floor(x/k)
        #      = -x * sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k)*floor(x/k)
        #      = sum_{k=2}^K f(k) * (floor(x/k) - x/k)
        #
        # So the constraint is: sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1.0001
        # Note: floor(x/k) - x/k is in [-1, 0), so this is a "negative" constraint
        # on the f(k) values.
        
        x_vals = np.arange(1, 10 * K + 1, dtype=np.float64)
        
        # Vectorized construction of A_ub
        # A_ub[i, j] = floor(x_vals[i] / k_vals[j]) - x_vals[i] / k_vals[j]
        x_col = x_vals[:, np.newaxis]  # (10K, 1)
        k_row = k_vals[np.newaxis, :]  # (1, K-1)
        
        A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10K, K-1)
        b_ub = np.full(10 * K, 1.0001)
        
        # Bounds: f(k) in [-10, 10]
        bounds = [(-10.0, 10.0)] * n_vars
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                # Compute the objective value: -sum f(k) log(k)/k
                obj_val = -np.sum(f_vals * c)
                
                if obj_val > best_score:
                    # Construct the result
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = 0.0  # f(1) placeholder, will be adjusted by evaluator
                    values[1:] = f_vals
                    
                    best_result = (domain, values)
                    best_score = obj_val
                    
                    # Print progress for debugging
                    print(f"K={K}: score={obj_val:.6f}, elapsed={time.time()-start_time:.1f}s")
                    
        except Exception as e:
            print(f"K={K}: LP failed with {e}")
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple mu-based approach
    K = 100
    mu = np.zeros(K + 1, dtype=int)
    mu[1] = 1
    for i in range(2, K + 1):
        if mu[i] == 0:  # prime
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
                if j % (i * i) == 0:
                    mu[j] = 0
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = np.zeros(K, dtype=np.float64)
    values[1:] = mu[2:K + 1].astype(np.float64)
    return domain, values

