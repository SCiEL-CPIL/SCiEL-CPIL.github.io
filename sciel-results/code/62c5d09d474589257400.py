"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: Use NumPy for fast sigma sieve and vectorized grouping.
    # Try progressively larger K, keeping the best result found.
    
    def try_sieve(K):
        nonlocal best_M, best_k, best_count
        # 1. Compute Sigma using NumPy
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            sigma[d::d] += d
        
        if time.time() > deadline - 25:
            return False

        # 2. Compute M values
        ks_range = np.arange(1, K + 1, dtype=np.int64)
        Ms = ks_range * sigma[1:]
        
        if time.time() > deadline - 15:
            return False

        # 3. Group by M using np.unique
        unique_Ms, inverse, counts = np.unique(Ms, return_inverse=True, return_counts=True)
        
        # Find the index of the maximum count
        max_count_idx = np.argmax(counts)
        max_count = counts[max_count_idx]
        
        if max_count > best_count:
            best_M = int(unique_Ms[max_count_idx])
            k_indices = np.where(Ms == best_M)[0]
            best_k = [int(x + 1) for x in k_indices]
            best_count = max_count
        
        # Free memory
        del sigma, Ms, unique_Ms, inverse, counts, ks_range
        return True
    
    # Try increasing K values, stop if time is running out
    for K in [10000000, 20000000, 40000000, 60000000]:
        if time.time() > deadline - 40:
            break
        try:
            ok = try_sieve(K)
            if not ok:
                break
        except MemoryError:
            # Try next smaller size
            continue
        except Exception:
            break
    
    # Final safety: ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
