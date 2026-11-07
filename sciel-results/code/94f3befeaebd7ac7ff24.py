import math
import time
from collections import defaultdict
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def compute_sigma_numpy(N):
        """Compute sigma(n) for all n up to N using NumPy vectorized SPF sieve."""
        # Create SPF array
        spf = np.arange(N + 1, dtype=np.int64)
        spf[0] = 0
        spf[1] = 1
        
        # Sieve for smallest prime factors
        limit = int(math.isqrt(N))
        for i in range(2, limit + 1):
            if spf[i] == i:  # i is prime
                start = i * i
                # Vectorized: mark multiples
                indices = np.arange(start, N + 1, i)
                mask = spf[indices] == indices
                spf[indices[mask]] = i
        
        # Compute sigma multiplicatively
        sigma = np.ones(N + 1, dtype=np.int64)
        sigma[0] = 0
        sigma[1] = 1
        
        for i in range(2, N + 1):
            p = spf[i]
            m = i // p
            if m % p == 0:
                # Find highest power of p dividing i
                power = p
                while m % p == 0:
                    m //= p
                    power *= p
                sigma[i] = sigma[m] * (power * p - 1) // (p - 1)
            else:
                sigma[i] = sigma[m] * (1 + p)
        
        return sigma
    
    # Start with a reasonable N, can increase if time permits
    N = 5_000_000
    
    t0 = time.time()
    
    # Compute sigma using NumPy-accelerated SPF
    sigma = compute_sigma_numpy(N)
    
    t1 = time.time()
    elapsed = t1 - t0
    
    # Find M with most solutions
    M_to_ks = defaultdict(list)
    best_M = 333312
    best_sols = [336, 372, 434]
    best_count = len(best_sols)
    
    for k in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        M_val = k * sigma[k]
        if M_val <= 10**18:
            M_to_ks[M_val].append(k)
            if len(M_to_ks[M_val]) > best_count:
                best_count = len(M_to_ks[M_val])
                best_sols = M_to_ks[M_val][:]
                best_M = M_val
    
    # Final verification
    best_sols = list(set(best_sols))
    limit = math.isqrt(best_M)
    valid_sols = []
    for k in best_sols:
        if 1 <= k <= limit and k <= N:
            if k * sigma[k] == best_M:
                valid_sols.append(k)
    
    if not valid_sols:
        best_M = 333312
        valid_sols = [336, 372, 434]
    
    return {'M': int(best_M), 'k': [int(x) for x in valid_sols]}
