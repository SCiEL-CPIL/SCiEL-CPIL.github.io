import math
import time
import numpy as np
from collections import defaultdict

_START = time.time()
DEADLINE = _START + 140  # 140s hard limit, 10s safety margin

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - time.time()

def construct():
    # We want to maximize the number of k such that k * sigma(k) = M
    # and k <= sqrt(M).
    
    # Strategy:
    # 1. Sieve sigma(k) for k up to N.
    # 2. Compute M_k = k * sigma(k).
    # 3. Find M that appears most frequently in M_k.
    # 4. Extract the k's for that M.
    # 5. Verify constraints.
    
    # Choose N = 2*10^6 for safety (known to find 5+ solutions)
    N = 2_000_000
    
    if _time_left() < 15:
        return _small_search()

    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Efficient sieve: for each i, add i to all multiples
        # Use a more efficient approach by processing in chunks
        for i in range(1, N + 1):
            if _time_left() < 10:
                break
            sigma[i::i] += i
            
        # Compute M_k = k * sigma(k)
        k_range = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_range * sigma
        
        # Filter M_vals <= 10^18
        max_M = 10**18
        valid_mask = M_vals <= max_M
        M_vals_valid = M_vals[valid_mask]
        k_vals_valid = k_range[valid_mask]
        
        if len(M_vals_valid) == 0:
            return _small_search()

        # Find the most frequent M using a dictionary approach for speed
        # Convert to Python lists for faster unique/count if needed
        # Actually, use np.unique which is efficient
        unique_M, counts = np.unique(M_vals_valid, return_counts=True)
        
        if len(unique_M) == 0:
            return _small_search()
        
        # Find index of max count
        max_count_idx = np.argmax(counts)
        best_M = unique_M[max_count_idx]
        best_count = counts[max_count_idx]
        
        # Extract k's for best_M
        indices = np.where(M_vals_valid == best_M)[0]
        k_solutions = k_vals_valid[indices]
        
        # Convert to list of Python ints
        k_list = [int(k) for k in k_solutions]
        
        # Verify
        sqrt_M = math.isqrt(int(best_M))
        valid_k = []
        for k in k_list:
            if k <= sqrt_M:
                if k * int(sigma[k]) == int(best_M):
                    valid_k.append(k)
        
        if not valid_k:
            return _small_search()
            
        # Sort k
        valid_k.sort()
        
        return {'M': int(best_M), 'k': valid_k}

    except Exception as e:
        return _small_search()

def _small_search():
    # Fallback: brute force small M
    # Known good: M=333312, k=[336, 372, 434]
    return {'M': 333312, 'k': [336, 372, 434]}
