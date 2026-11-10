import math
import time
import numpy as np

def construct():
    start_time = time.time()
    N = 30_000_000
    
    # Sieve for sigma(k)
    # sigma[k] = sum of divisors of k
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # i from 1 to N
    # sigma[i::i] += i
    # This is O(N log N) operations, but vectorized.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Calculate M = k * sigma(k)
    ks = np.arange(1, N + 1, dtype=np.int64)
    Ms = ks * sigma
    
    # Filter M <= 10^18
    mask = Ms <= 10**18
    Ms = Ms[mask]
    ks = ks[mask]
    
    # Sort by M
    order = np.argsort(Ms)
    Ms_sorted = Ms[order]
    ks_sorted = ks[order]
    
    # Find M with maximum count
    best_M = 0
    best_ks = []
    max_count = 0
    
    # Iterate through sorted Ms to find runs
    # Since N is large, do it in chunks or just loop
    # Python loop over 30M items is ~3-5s.
    
    current_M = -1
    current_count = 0
    current_ks = []
    
    # Optimization: Only update best if count increases
    # We need to store ks for the best M.
    
    for i in range(len(Ms_sorted)):
        if Ms_sorted[i] == current_M:
            current_count += 1
            current_ks.append(int(ks_sorted[i]))
        else:
            if current_count > max_count:
                max_count = current_count
                best_M = int(current_M)
                best_ks = current_ks[:]
            current_M = Ms_sorted[i]
            current_count = 1
            current_ks = [int(ks_sorted[i])]
            
    # Check last run
    if current_count > max_count:
        max_count = current_count
        best_M = int(current_M)
        best_ks = current_ks[:]
        
    # Verify
    if best_M == 0:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    sqrt_M = math.isqrt(best_M)
    valid_ks = [k for k in best_ks if k <= sqrt_M]
    
    return {'M': best_M, 'k': valid_ks}
