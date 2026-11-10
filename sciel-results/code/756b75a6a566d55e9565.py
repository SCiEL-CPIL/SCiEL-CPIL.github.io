"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _isqrt(x):
    return math.isqrt(x)

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Known fallback solution
    fallback_M = 333312
    fallback_k = [336, 372, 434]
    
    best_M = fallback_M
    best_k = list(fallback_k)
    best_count = len(best_k)
    
    # Try progressively larger N values
    # Keep N values conservative to avoid timeout
    N_values = [10**6, 2*10**6, 5*10**6, 10**7]
    
    for N in N_values:
        if time.time() - start_time > deadline - 30:
            break
        
        # Compute sigma(1..N) using NumPy sieve
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
        
        # Vectorized computation of M = k * sigma(k) for k=1..N
        ks = np.arange(1, N + 1, dtype=np.int64)
        sigmas = sigma[1:N+1]
        Ms = ks * sigmas
        
        # Filter: M <= 10^18
        valid_mask = Ms <= 10**18
        valid_ks = ks[valid_mask]
        valid_Ms = Ms[valid_mask]
        
        if len(valid_Ms) == 0:
            continue
        
        # Find M with most preimages using np.unique
        unique_Ms, counts = np.unique(valid_Ms, return_counts=True)
        max_idx = int(np.argmax(counts))
        max_count = int(counts[max_idx])
        
        if max_count > best_count:
            max_M = int(unique_Ms[max_idx])
            mask = (valid_Ms == max_M)
            best_k = [int(k) for k in valid_ks[mask]]
            best_M = max_M
            best_count = max_count
        
        if time.time() - start_time > deadline - 15:
            break
        
        if best_count >= 100:
            break
    
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    for k in best_k:
        if k > _isqrt(best_M):
            return {'M': fallback_M, 'k': list(fallback_k)}
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
