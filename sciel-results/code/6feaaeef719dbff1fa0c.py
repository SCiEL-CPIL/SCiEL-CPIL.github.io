import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 1
    best_s = 1

    def compute_phi_small(L, R):
        """Compute phi for a small range [L, R] by factoring each number."""
        n = R - L + 1
        phi = np.zeros(n, dtype=np.int64)
        for i in range(n):
            num = L + i
            temp = num
            p_val = num
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    while temp % p == 0:
                        temp //= p
                    p_val = p_val // p * (p - 1)
            if temp > 1:
                p_val = p_val // temp * (temp - 1)
            phi[i] = p_val
        return phi

    def max_distinct_run(phi):
        """Find longest run of distinct values in phi array."""
        n = len(phi)
        if n == 0:
            return 0, 0
        
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        # Use a list for phi to speed up access?
        # phi is numpy array. Accessing phi[i] is slower than list[i].
        # But converting to list takes time.
        # For n=10^7, list conversion is fast (~0.1s).
        phi_list = phi.tolist()
        
        for right in range(n):
            val = phi_list[right]
            if val in last_seen:
                if last_seen[val] >= left:
                    left = last_seen[val] + 1
            last_seen[val] = right
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
        
        return max_len, max_start

    def process_segment(L, R):
        nonlocal best_k, best_s
        if R > 10**12:
            R = 10**12
        if L > R:
            return
        n = R - L + 1
        phi = np.arange(L, R+1, dtype=np.int64)
        temp = np.arange(L, R+1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            mask = temp % p == 0
            if np.any(mask):
                phi[mask] = phi[mask] // p * (p - 1)
                temp[mask] = temp[mask] // p
        
        mask = temp > 1
        if np.any(mask):
            q = temp[mask]
            phi[mask] = phi[mask] // q * (q - 1)
        
        max_len, max_start = max_distinct_run(phi)
        
        if max_len > best_k:
            best_k = max_len
            best_s = L + max_start
            
        return best_k, best_s

    # 1. Initial scan of [1, 10^6] to get a baseline quickly
    # 10^6 is fast enough for the Python loop in max_distinct_run.
    if time.time() < deadline - 10:
        process_segment(1, 10**6)
        
    # 2. Expand scan to [1, 10^7] if time permits
    if time.time() < deadline - 5:
        process_segment(10**6 + 1, 10**7)
        
    # 3. Random sampling in [10^7, 10^12]
    # We look for runs longer than best_k.
    # Window size for sampling: best_k + 100
    # Number of samples: as many as time allows.
    
    if time.time() < deadline - 2:
        window_size = best_k + 100
        if window_size > 200:
            window_size = 200
            
        # Use random sampling
        # We need to be careful with time.
        # compute_phi_small for 200 numbers is very fast (~ms).
        # max_distinct_run for 200 numbers is instant.
        
        # Let's do 5000 samples
        num_samples = 5000
        # Ensure we don't exceed time
        # Estimate time per sample: 0.5 ms?
        # 5000 * 0.5 ms = 2.5 s.
        
        for _ in range(num_samples):
            if time.time() > deadline - 2:
                break
            
            # Pick random start in [10^7, 10^12 - window_size]
            # Uniform distribution in log space? Or linear?
            # Linear is fine for now, but log space might be better for "structure".
            # Let's use linear for simplicity, but biased towards larger numbers?
            # Actually, phi collisions might be more frequent for smaller numbers?
            # Or less?
            # Let's just sample uniformly.
            
            s = np.random.randint(10**7, 10**12 - window_size)
            
            # Compute phi for [s, s + window_size - 1]
            phi_seg = compute_phi_small(s, s + window_size - 1)
            
            # Find max distinct run
            max_len, max_start = max_distinct_run(phi_seg)
            
            if max_len > best_k:
                best_k = max_len
                best_s = s + max_start
                
                # If we found a very long run, we might want to extend it?
                # Or just continue sampling.
                # Extending requires checking neighbors.
                # Let's just update best.

    return {'start': best_s, 'length': best_k}
