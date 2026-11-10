import math
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try increasing sieve sizes - start with larger to find better solutions
    sieve_sizes = [50_000_000, 35_000_000, 20_000_000]
    
    for N in sieve_sizes:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        try:
            # Compute sigma using divisor-sum sieve
            sigma = np.zeros(N + 1, dtype=np.int64)
            sig = sigma
            for d in range(1, N + 1):
                sig[d::d] += d
                
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 25:
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
                
            # Find best M using sort + run-length encoding (faster than np.unique for large arrays)
            sorted_Ms = np.sort(Ms_valid)
            # Find where values change
            diff = np.diff(sorted_Ms)
            change_idx = np.where(diff != 0)[0] + 1
            # Start indices of each run
            start_indices = np.concatenate(([0], change_idx))
            end_indices = np.concatenate((change_idx, [len(sorted_Ms)]))
            # Count of each run
            counts = end_indices - start_indices
            
            max_idx = int(np.argmax(counts))
            max_count = int(counts[max_idx])
            
            if max_count > best_count:
                best_M = int(sorted_Ms[start_indices[max_idx]])
                # Get the k values for this M
                mask = (Ms_valid == best_M)
                best_ks = sorted(set(int(x) for x in ks_valid[mask]))
                best_count = max_count
                
        except (MemoryError, Exception) as e:
            continue
    
    # Ensure types are correct
    return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
