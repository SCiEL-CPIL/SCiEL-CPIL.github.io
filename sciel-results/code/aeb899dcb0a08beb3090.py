import math
import time
import sys
import numpy as np

def _sigma_sieve(limit):
    """Compute sigma(n) for all n in 1..limit using a divisor-sum sieve."""
    sigma = np.zeros(limit + 1, dtype=np.int64)
    # d=1 adds 1 to all
    if limit >= 1:
        sigma[1:] += 1
    for d in range(2, limit + 1):
        sigma[d::d] += d
    return sigma

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try increasing sieve sizes
    # 50M might take too long or OOM. 30M is a good middle ground.
    # The current program uses 20M.
    # Let's try 30M first, then 50M if time permits? No, sequential is risky.
    # Let's try 40M.
    
    sieve_sizes = [40_000_000, 25_000_000, 15_000_000]
    
    for N in sieve_sizes:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        try:
            # Check memory roughly? 40M * 8 bytes = 320MB. Should be fine.
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            # Sieve
            # Optimization: local vars
            sig = sigma
            for d in range(1, N + 1):
                sig[d::d] += d
                
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 15:
                continue
                
            # Compute M = k * sigma(k)
            ks = np.arange(1, N + 1, dtype=np.int64)
            Ms = ks * sig[1:N+1]
            
            # Filter M <= 10^18
            valid = Ms <= 10**18
            Ms_valid = Ms[valid]
            ks_valid = ks[valid]
            
            if len(Ms_valid) == 0:
                continue
                
            # Find best M
            # np.unique is O(N log N)
            unique_Ms, inverse, counts = np.unique(Ms_valid, return_inverse=True, return_counts=True)
            
            if len(unique_Ms) == 0:
                continue
                
            max_idx = int(np.argmax(counts))
            max_count = int(counts[max_idx])
            
            if max_count > best_count:
                best_M = int(unique_Ms[max_idx])
                mask = (inverse == max_idx)
                best_ks = sorted(set(int(x) for x in ks_valid[mask]))
                best_count = max_count
                
                # If we found a very good result, we might stop early?
                # But maybe a larger N would find even more?
                # Let's continue to next N only if we have time.
                
        except (MemoryError, Exception) as e:
            # Fallback to smaller N or skip
            continue

    # Ensure types are correct
    return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
