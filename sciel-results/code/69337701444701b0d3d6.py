import numpy as np
import time
import sys
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 550  # Safety margin (600s limit)
    
    # Initial fallback from the seed program
    # Recalculate the raw score 'a' for the seed to use as baseline
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # Calculate raw score a = -sum f(k) log(k)/k
    ks = seed_domain.astype(np.float64)
    fs = seed_values
    # log(1) is 0, so term 1 is 0
    best_raw_score = -np.sum(fs * np.log(ks) / ks)
    
    best_domain = seed_domain
    best_values = seed_values
    
    # Try a range of K values. 
    # Larger K allows for a better approximation of the optimal function.
    # We prioritize larger K if time permits.
    # Memory estimate: K=3000 -> 30000 x 2999 floats ~ 720MB. 
    # K=5000 -> 50000 x 4999 floats ~ 2GB. Might be too much.
    K_candidates = [1000, 2000, 3000]
    
    for K in K_candidates:
        # Check time
        if time.time() - start_time > timeout:
            break
            
        try:
            # Variables f_2 ... f_K
            n_vars = K - 1
            indices = np.arange(2, K + 1, dtype=np.float64)
            
            # Objective: maximize -sum f(k) log(k)/k
            # linprog minimizes, so we minimize sum f(k) log(k)/k
            c = np.log(indices) / indices
            
            # Constraints: M(x) = -sum f(k) {x/k} <= 1.0001 for x in 1..10K
            # A_ub @ f <= b_ub
            # A_ub[x, k] = -{x/k}
            
            x_vals = np.arange(1, 10 * K + 1, dtype=np.float64)
            
            # Vectorized calculation of fractional parts
            # Shape: (10K, K-1)
            
            # To reduce memory peak, we can compute in chunks if necessary, 
            # but let's try direct first for speed.
            
            # x_vals[:, None] / indices[None, :]
            # This creates a large temporary array.
            
            # If K is large, this might fail.
            
            quotients = x_vals[:, None] / indices[None, :]
            frac_parts = quotients - np.floor(quotients)
            A_ub = -frac_parts
            
            b_ub = np.full(10 * K, 1.0001)
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve
            # Use 'highs' for speed and robustness
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0:
                f_vals = res.x
                
                # Verify feasibility
                # M(x) = -sum f(k) {x/k}
                # We already have frac_parts.
                M_vals = -np.dot(frac_parts, f_vals)
                
                if np.max(M_vals) <= 1.0001 + 1e-6:
                    # Calculate score
                    score = -np.sum(f_vals * c)
                    if score > best_raw_score:
                        best_raw_score = score
                        # Construct domain and values
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[0] = 0.0 # f(1) orig, evaluator adjusts
                        values[1:] = f_vals
                        best_domain = domain
                        best_values = values
                        # print(f"New best raw score: {best_raw_score} for K={K}", file=sys.stderr)
                        
        except MemoryError:
            print(f"Memory error for K={K}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
            
    return best_domain, best_values
