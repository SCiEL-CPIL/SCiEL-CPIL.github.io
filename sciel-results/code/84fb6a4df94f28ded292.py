import math
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Use int32 for sigma to reduce memory and improve cache performance
    # Max sigma(k) for k <= 100M is well under 2^31 (~2.1B)
    # sigma(k) <= k * log(log(k)) + O(k), for k=100M this is ~300M
    
    sieve_sizes = [100_000_000, 60_000_000, 30_000_000, 15_000_000]
    
    for N in sieve_sizes:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        try:
            sigma = np.zeros(N + 1, dtype=np.int32)
            
            # Sieve: for each d, add d to all multiples of d
            for d in range(1, N + 1):
                sigma[d::d] += d
            
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 20:
                continue
            
            # Compute M = k * sigma(k)
            ks = np.arange(1, N + 1, dtype=np.int64)
            # Convert sigma to int64 to avoid overflow in multiplication
            Ms = ks * sigma[1:N+1].astype(np.int64)
            
            # Filter M <= 10^18
            valid = Ms <= 10**18
            Ms_valid = Ms[valid]
            ks_valid = ks[valid]
            
            if len(Ms_valid) == 0:
                continue
            
            # Find M with most occurrences
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
                
                # Stop early if we found a very good result
                if best_count >= 20:
                    break
                    
        except (MemoryError, Exception) as e:
            continue
    
    # Ensure types are correct
    return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
