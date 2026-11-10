import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600
    
    MAX_K = 200000
    
    # Linear Sieve for Mobius
    mu = [0] * (MAX_K + 1)
    mu[1] = 1
    is_comp = [False] * (MAX_K + 1)
    primes = []
    
    for i in range(2, MAX_K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > MAX_K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    best_score = -float('inf')
    best_K = 1
    
    sum_inv_mu = 0.0
    sum_log_mu = 0.0
    
    # Precompute logs? math.log is fast enough.
    
    for k in range(1, MAX_K + 1):
        v = mu[k]
        sum_inv_mu += v / k
        sum_log_mu += v * math.log(k) / k
        
        current_score = -sum_log_mu
        
        # Feasibility check
        # M(x) = 1 - x * sum_inv_mu
        # We need M(x) <= 1.0001 for all x in [1, 10*k]
        
        # If sum_inv_mu > 0, M(x) decreases. Max at x=1.
        # If sum_inv_mu < 0, M(x) increases. Max at x=10*k.
        # If sum_inv_mu == 0, M(x) = 1.
        
        if sum_inv_mu >= 0:
             # Max at x=1
             max_M = 1.0 - sum_inv_mu
        else:
             # Max at x=10*k
             max_M = 1.0 - 10.0 * k * sum_inv_mu
             
        if max_M <= 1.0001:
            if current_score > best_score:
                best_score = current_score
                best_K = k
                
        if time.time() > deadline - 10:
            break

    domain = list(range(1, best_K + 1))
    values = mu[1:best_K + 1]
    
    # Convert to numpy arrays as required
    import numpy as np
    return np.array(domain, dtype=np.int64), np.array(values, dtype=np.float64)
