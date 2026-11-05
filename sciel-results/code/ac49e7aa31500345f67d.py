import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # Leave margin for overhead
    MAX_S = 10**12

    # 1. Generate primes up to 10^6 for trial division / segmented sieve
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    # 2. Segmented Phi Computation using NumPy
    def compute_phi_segment(L, R):
        size = R - L + 1
        phi_vals = np.arange(L, R + 1, dtype=np.int64)
        rem = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            # Start of multiples of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            # Vectorized: process all multiples of p at once
            indices = np.arange(start, R + 1, p) - L
            phi_vals[indices] = phi_vals[indices] // p * (p - 1)
            # Remove all factors of p from rem
            mask = indices
            while True:
                vals = rem[mask]
                divisible = vals % p == 0
                if not np.any(divisible):
                    break
                new_mask = mask[divisible]
                rem[new_mask] = rem[new_mask] // p
                mask = new_mask
        
        # Handle remaining prime factors > sqrt(R)
        remaining = rem > 1
        if np.any(remaining):
            idxs = np.where(remaining)[0]
            phi_vals[idxs] = phi_vals[idxs] // rem[idxs] * (rem[idxs] - 1)
                
        return phi_vals.tolist()

    # 3. Search strategy
    best_k = 1
    best_s = 1
    
    # We will scan blocks of size B.
    # With NumPy vectorization, we can use larger blocks.
    block_size = 200000
    current_end = MAX_S
    
    # We need to be careful with the time limit.
    # Let's reserve 5 seconds for the final return/cleanup.
    
    while current_end > 1:
        if time.time() > DEADLINE:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        # Compute phis for this block using segmented sieve
        phis = compute_phi_segment(block_start, current_end)
        
        # Find longest run of distinct values
        # Two-pointer technique
        seen_vals = {}  # value -> index
        l = 0
        max_len = 0
        max_start_idx = 0
        
        for r in range(len(phis)):
            v = phis[r]
            if v in seen_vals:
                prev_idx = seen_vals[v]
                if prev_idx >= l:
                    l = prev_idx + 1
            seen_vals[v] = r
            
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start_idx = l
            
            # Cap at 200
            if max_len >= 200:
                best_k = 200
                best_s = block_start + max_start_idx
                return {'start': best_s, 'length': best_k}
        
        if max_len > best_k:
            best_k = max_len
            best_s = block_start + max_start_idx
            
        # Move to previous block
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
