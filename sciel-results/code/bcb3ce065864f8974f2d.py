import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 500  # Safety margin
    
    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        if K >= 1:
            mu[1] = 1
        primes = []
        is_comp = np.zeros(K + 1, dtype=bool)
        
        for i in range(2, K + 1):
            if not is_comp[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_comp[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility(K, mu):
        if K < 1:
            return False, None, None
        
        ks = np.arange(1, K + 1)
        # f(k) = -mu(k) for k > 1
        fs = -mu[1:K+1].astype(np.float64)
        
        # Calculate f(1) such that sum f(k)/k = 0
        # sum_{k=1}^K f(k)/k = f(1)/1 + sum_{k=2}^K f(k)/k = 0
        # f(1) = - sum_{k=2}^K f(k)/k
        if K > 1:
            sum_fk_over_k = np.sum(fs[1:] / ks[1:])
            fs[0] = -sum_fk_over_k
        else:
            fs[0] = 0.0
            
        # Clip values
        fs = np.clip(fs, -10, 10)
        
        # Check M(x) for x in 1..10K
        max_x = 10 * K
        
        # Memory check: max_x * K elements
        # K=2000 -> 20000 * 2000 = 40M floats = 320MB.
        # K=1000 -> 10000 * 1000 = 10M floats = 80MB.
        # Let's try K=1000 first. If feasible, return.
        
        if K > 1500:
             # If K is too large, skip to avoid memory error in this function
             # But we call with specific K.
             pass

        xs = np.arange(1, max_x + 1)
        
        # Vectorized computation
        # To save memory, we can do it in chunks of x
        # But for K=1000, it's fine.
        
        k_vec = np.arange(1, K + 1)
        
        # We need M(x) for all x.
        # M(x) = sum f(k) * floor(x/k)
        
        # If K is small, do full matrix.
        if K <= 1500:
            x_vec = np.arange(1, max_x + 1)[:, None]
            floors = np.floor(x_vec / k_vec[None, :])
            M_vals = np.sum(fs[None, :] * floors, axis=1)
            max_M = np.max(M_vals)
        else:
            # Fallback for larger K if we ever get there
            M_vals = np.zeros(max_x + 1)
            for k in range(1, K + 1):
                if fs[k-1] == 0:
                    continue
                M_vals[1:] += fs[k-1] * np.floor(np.arange(1, max_x + 1) / k)
            max_M = np.max(M_vals)
            
        if max_M <= 1.0001:
            return True, ks, fs
        else:
            return False, ks, fs

    # Try K values
    # We want a good score, which increases with K.
    # But K=1000 is safe for memory.
    for K in [1000, 500, 200, 100]:
        if time.time() > deadline:
            break
        mu = compute_mobius(K)
        feasible, domain, values = check_feasibility(K, mu)
        if feasible:
            return domain, values
            
    # Fallback: trivial solution
    domain = np.array([1], dtype=np.int64)
    values = np.array([0.0], dtype=np.float64)
    return domain, values
