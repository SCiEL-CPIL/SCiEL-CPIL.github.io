import numpy as np
import time
import math

def construct():
    # Deadline: 115 seconds (5s margin)
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Segmented Phi Calculation
    def compute_phi_segment(L, R, primes):
        n = R - L + 1
        rem = np.arange(L, R + 1, dtype=np.int64)
        phi = np.arange(L, R + 1, dtype=np.int64)
        
        # Process primes
        for p in primes:
            if p * p > R:
                break
            
            # Find first multiple of p in [L, R]
            m0 = ((L + p - 1) // p) * p
            if m0 > R:
                continue
            
            pos = m0 - L
            idx = np.arange(pos, n, p)
            
            if len(idx) == 0:
                continue
            
            phi[idx] = (phi[idx] // p) * (p - 1)
            rem[idx] = rem[idx] // p
            
            while True:
                mask = (rem[idx] % p == 0)
                if not np.any(mask):
                    break
                rem[idx] = rem[idx] // p
        
        mask = rem > 1
        if np.any(mask):
            phi[mask] = (phi[mask] // rem[mask]) * (rem[mask] - 1)
        
        return phi

    # 3. Sliding Window to find max distinct run
    SEG_SIZE = 10**6
    TAIL_SIZE = 5000
    
    best_len = 1
    best_start = 1
    
    tail_vals = []
    tail_start = 1
    
    L = 1
    while L <= 10**12:
        if time.time() > deadline:
            break
        
        R = min(L + SEG_SIZE - 1, 10**12)
        
        phi_vals = compute_phi_segment(L, R, primes)
        
        # Combine with tail
        # tail_vals corresponds to [tail_start, tail_start + len(tail_vals) - 1]
        # We want to scan the union.
        
        if len(tail_vals) > TAIL_SIZE:
             # This shouldn't happen if we set TAIL_SIZE correctly in previous step
             pass
             
        combined_vals = tail_vals + list(phi_vals)
        combined_start = tail_start
        
        seen = set()
        left = 0
        
        for right in range(len(combined_vals)):
            v = combined_vals[right]
            while v in seen:
                seen.remove(combined_vals[left])
                left += 1
            seen.add(v)
            
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = combined_start + left
                
        # Update tail
        new_tail_len = min(TAIL_SIZE, len(phi_vals))
        tail_vals = list(phi_vals[-new_tail_len:])
        tail_start = R - new_tail_len + 1
        
        L = R + 1
        
    return {'start': best_start, 'length': best_len}
