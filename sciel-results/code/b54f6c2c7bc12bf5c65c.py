"""Rewritten program: sieve-based enumeration to find M with many preimages k where k*sigma(k)=M."""
import math
import time
import sys

def _sigma_sieve(limit):
    """Compute sigma(n) for all n in 1..limit using a divisor-sum sieve.
    Returns a list sigma where sigma[i] = sum of divisors of i."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def _find_best_from_arrays(Ms, ks):
    """Given arrays Ms and ks (same length, filtered valid), find the M with most distinct k.
    Returns (best_M, sorted_list_of_k) or (None, [])."""
    try:
        import numpy as np
    except ImportError:
        return None, []
    
    if len(Ms) == 0:
        return None, []
    
    # Use numpy unique to group: unique_Ms, inverse, counts
    unique_Ms, inverse, counts = np.unique(Ms, return_inverse=True, return_counts=True)
    
    if len(unique_Ms) == 0:
        return None, []
    
    # Find the index of the M with maximum count
    max_idx = int(np.argmax(counts))
    best_M = int(unique_Ms[max_idx])
    max_count = int(counts[max_idx])
    
    if max_count < 2:
        return best_M, []
    
    # Extract all k values that map to best_M
    mask = (inverse == max_idx)
    best_ks = ks[mask]
    best_ks = sorted(set(int(x) for x in best_ks))
    
    return best_M, best_ks

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    try:
        import numpy as np
        HAVE_NUMPY = True
    except ImportError:
        HAVE_NUMPY = False
    
    best_result = None  # (M, k_list)
    
    if HAVE_NUMPY:
        # Use numpy for faster sieve
        # Try multiple sieve sizes, starting large
        sieve_sizes = [20_000_000, 10_000_000, 5_000_000, 3_000_000]
        
        for N in sieve_sizes:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                sigma = np.zeros(N + 1, dtype=np.int64)
                for d in range(1, N + 1):
                    sigma[d::d] += d
                
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - 20:
                    continue
                
                # Compute M = k * sigma(k) for k = 1..N
                ks = np.arange(1, N + 1, dtype=np.int64)
                Ms = ks * sigma[1:N+1]
                
                # Filter: M <= 10^18
                valid = Ms <= 10**18
                Ms_valid = Ms[valid]
                ks_valid = ks[valid]
                
                if len(Ms_valid) == 0:
                    continue
                
                # Find best M using numpy unique (much faster than Python dict)
                best_M, best_ks = _find_best_from_arrays(Ms_valid, ks_valid)
                
                if best_M is not None and len(best_ks) >= 2:
                    if best_result is None or len(best_ks) > len(best_result[1]):
                        best_result = (best_M, best_ks)
                    
                    # If we have a good result and time is running out, return it
                    if time.time() - start_time > DEADLINE - 10:
                        break
                        
            except (MemoryError, Exception):
                continue
    
    # Fallback: pure Python sieve with smaller N
    if best_result is None or len(best_result[1]) < 3:
        for N in [3_000_000, 2_000_000, 1_000_000]:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                sigma = _sigma_sieve(N)
                
                if time.time() - start_time > DEADLINE - 20:
                    continue
                
                from collections import defaultdict
                groups = defaultdict(list)
                for k in range(1, N + 1):
                    M = k * sigma[k]
                    if M <= 10**18:
                        groups[M].append(k)
                
                best_M = None
                best_ks = []
                for m, k_list in groups.items():
                    if len(k_list) > len(best_ks):
                        best_M = m
                        best_ks = k_list
                
                if best_M is not None:
                    best_ks = sorted(set(best_ks))
                    if len(best_ks) >= 2:
                        if best_result is None or len(best_ks) > len(best_result[1]):
                            best_result = (best_M, best_ks)
                        
            except (MemoryError, Exception):
                continue
    
    if best_result is not None and len(best_result[1]) >= 2:
        return {'M': int(best_result[0]), 'k': [int(x) for x in best_result[1]]}
    
    # Ultimate fallback: return the seed result
    return {'M': 333312, 'k': [336, 372, 434]}
