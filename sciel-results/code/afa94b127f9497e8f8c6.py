import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Step 1: Generate primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Segmented Totient Sieve
    def compute_phi_segment(S, W):
        vals = np.arange(S, S + W, dtype=np.int64)
        rem = vals.copy()
        
        sqrt_max = int(math.isqrt(S + W - 1))
        
        for p in primes:
            if p > sqrt_max:
                break
            
            first_multiple = ((S + p - 1) // p) * p
            if first_multiple >= S + W:
                continue
                
            indices = np.arange(first_multiple - S, W, p)
            
            vals[indices] = vals[indices] // p * (p - 1)
            
            idxs = indices
            while True:
                mask = rem[idxs] % p == 0
                if not np.any(mask):
                    break
                idxs = idxs[mask]
                rem[idxs] //= p
        
        mask_large = rem > 1
        if np.any(mask_large):
            q_vals = rem[mask_large]
            idx_large = np.where(mask_large)[0]
            vals[idx_large] = vals[idx_large] // q_vals * (q_vals - 1)
            
        return vals

    def find_longest_distinct_run(phi_vals, S):
        last_seen = {}
        left = 0
        max_len = 0
        best_start = S
        
        for right in range(len(phi_vals)):
            v = phi_vals[right]
            if v in last_seen and last_seen[v] >= left:
                left = last_seen[v] + 1
            last_seen[v] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                best_start = S + left
                
        return max_len, best_start

    best_k = 1
    best_s = 1
    
    # Step 3: Search
    W = 50000
    
    random.seed(42)
    
    # Use a mix of random and potentially systematic if time permits
    # For now, random segments
    
    # To avoid bias, we can pick S from log-uniform distribution
    # log S is uniform in [0, log(10^12)]
    # S = exp(uniform(0, 27.6))
    
    for i in range(300): # Try up to 300 segments
        if time.time() > deadline - 10:
            break
            
        # Log-uniform sampling
        log_S = random.uniform(0, math.log(10**12))
        S = int(math.exp(log_S))
        
        # Clamp S
        if S < 1:
            S = 1
        if S > 10**12 - W:
            S = 10**12 - W
            
        phi_vals = compute_phi_segment(S, W)
        
        k, s = find_longest_distinct_run(phi_vals, S)
        
        if k > best_k:
            best_k = k
            best_s = s
            
    return {'start': best_s, 'length': best_k}
