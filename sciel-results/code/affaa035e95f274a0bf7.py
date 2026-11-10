import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # seconds
    
    K = 200000
    
    # Compute Mobius function mu(k) for k=1..K using linear sieve
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    # Construct domain and values
    # Include only k where mu(k) != 0
    mask = mu != 0
    domain = np.nonzero(mask)[0]
    domain = domain[domain > 0]
    values = mu[domain].astype(np.float64)
    
    # Adjust f(1) to satisfy sum f(k)/k = 0
    # S = sum f(k)/k
    S = np.sum(values / domain)
    # Find index of 1
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S
    else:
        # Should not happen as mu(1)=1
        pass

    # Check feasibility
    # M(x) = sum_{n=1}^x g(n) where g(n) = sum_{k|n} f(k)
    # We need max M(x) for x in [1, 10K]
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # Update g using sieve method
    # For each k, add f(k) to g[k], g[2k], g[3k], ...
    for i in range(len(domain)):
        k = domain[i]
        f_k = values[i]
        # g[k::k] += f_k
        # Vectorized slice update
        g[k:max_x+1:k] += f_k
        
    # Prefix sum
    M = np.cumsum(g)
    
    # Check constraint
    if np.max(M) > 1.0001:
        # If infeasible, we might need to scale down or return a simpler function
        # But Mobius should be feasible.
        # For safety, if infeasible, return a trivial function?
        # Trivial function: f(1)=0, others 0. Score 0.
        # Better to try to fix?
        # We can scale all values by a factor c < 1.
        # M(x) becomes c * M(x).
        # Score becomes c * a.
        # If max M(x) is slightly > 1, scaling helps.
        
        max_val = np.max(M)
        if max_val > 0:
            scale = 1.0 / max_val
            # Add small margin
            scale *= 0.9999
            values = values * scale
            
            # Re-check
            g[:] = 0
            for i in range(len(domain)):
                k = domain[i]
                f_k = values[i]
                g[k:max_x+1:k] += f_k
            M = np.cumsum(g)
            if np.max(M) > 1.0001:
                # Still infeasible, return empty?
                # Or just return the scaled one and hope?
                # The contract says if infeasible, score 0.
                # So we must ensure feasibility.
                # If scaling doesn't work, maybe K is too large?
                # But scaling should always work if max_val > 0.
                pass

    return domain, values
