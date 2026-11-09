import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate primes up to 10^6 (sqrt(10^12))
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # 2. Define segment size and search strategy
    seg_size = 5 * 10**6
    num_segments_to_try = 20
    
    for seg_idx in range(num_segments_to_try):
        if time.time() > deadline:
            break
            
        L = seg_idx * seg_size + 1
        R = L + seg_size - 1
        
        if R > 10**12:
            R = 10**12
            L = max(1, R - seg_size + 1)
            
        size = R - L + 1
        if size <= 0:
            continue
            
        # Initialize phi values
        phi_vals = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply sieve updates
        for p in primes:
            if p > R:
                break
            first_mult = ((L + p - 1) // p) * p
            
            if first_mult > R:
                continue
                
            idx = np.arange(first_mult - L, size, p)
            phi_vals[idx] -= phi_vals[idx] // p
        
        # Find longest run of distinct values using last_occurrence array
        last_occurrence = np.full(size, -1, dtype=np.int64)
        
        if size > 1:
            sorted_indices = np.argsort(phi_vals, kind='stable')
            sorted_vals = phi_vals[sorted_indices]
            
            # For each i from 1 to size-1, if sorted_vals[i] == sorted_vals[i-1],
            # then last_occurrence[sorted_indices[i]] = sorted_indices[i-1]
            
            # Create mask for where values are equal to previous in sorted order
            equal_to_prev = sorted_vals[1:] == sorted_vals[:-1]
            
            # For all positions i (1-indexed in sorted array) where equal_to_prev[i-1] is True:
            #   last_occurrence[sorted_indices[i]] = sorted_indices[i-1]
            
            # Vectorized:
            # Indices in sorted array where value equals previous: 1..size-1 where equal_to_prev
            # We need to set last_occurrence at those original positions
            
            # The original positions to update: sorted_indices[1:][equal_to_prev]
            # The values to set: sorted_indices[:-1][equal_to_prev]
            
            target_indices = sorted_indices[1:][equal_to_prev]
            source_indices = sorted_indices[:-1][equal_to_prev]
            
            last_occurrence[target_indices] = source_indices
        
        # Sliding window to find longest run
        left = 0
        max_k_seg = 0
        max_start_seg = 0
        
        for right in range(size):
            prev = last_occurrence[right]
            if prev >= left:
                left = prev + 1
            
            cur_k = right - left + 1
            if cur_k > max_k_seg:
                max_k_seg = cur_k
                max_start_seg = left
                if max_k_seg == 200:
                    break
        
        if max_k_seg > best_k:
            best_k = max_k_seg
            best_s = L + max_start_seg
            
            if best_k == 200:
                return {'start': best_s, 'length': best_k}

    return {'start': best_s, 'length': best_k}
