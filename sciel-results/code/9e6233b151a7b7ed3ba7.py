import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 1
    best_s = 1
    
    # Function to compute phi for a segment and find best run
    def process_segment(L, S):
        nonlocal best_k, best_s
        R = L + S - 1
        phi_arr = np.arange(L, R+1, dtype=np.int64)
        
        sqrt_R = math.isqrt(R)
        for p in primes:
            if p > sqrt_R:
                break
            start = L - (L % p)
            if start < L: start += p
            if start > R: continue
            indices = np.arange(start, R+1, p)
            phi_arr[indices - L] -= phi_arr[indices - L] // p
            
        mask = phi_arr > 1
        if np.any(mask):
            phi_arr[mask] -= phi_arr[mask] // phi_arr[mask]
            
        # Sliding window scan
        phi_list = phi_arr.tolist()
        n = len(phi_list)
        if n == 0: return
        
        freq = {}
        dup_count = 0
        current_k = 0
        
        # Initialize window of size 1
        # We want max k such that phi[s..s+k-1] distinct.
        # Sliding window of size k.
        # We can expand the window as much as possible.
        
        # Alternative:
        # For each right endpoint j, find the smallest left endpoint i such that [i, j] is distinct.
        # Then the run length is j - i + 1.
        
        left = 0
        for right in range(n):
            val = phi_list[right]
            freq[val] = freq.get(val, 0) + 1
            if freq[val] == 2:
                dup_count += 1
            
            # Shrink window if duplicates exist
            while dup_count > 0:
                left_val = phi_list[left]
                freq[left_val] -= 1
                if freq[left_val] == 1:
                    dup_count -= 1
                left += 1
                
            # Now [left, right] is distinct
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_s = L + left
            if best_k == 200:
                return

    # Initial check
    process_segment(1, 100000)
    
    # Random sampling
    S = 5000000
    random.seed(42)
    
    while time.time() - start_time < deadline - 5:
        max_L = 10**12 - S
        if max_L < 1: break
        L = random.randint(1, max_L)
        process_segment(L, S)
        if best_k == 200:
            break
            
    return {'start': best_s, 'length': best_k}
