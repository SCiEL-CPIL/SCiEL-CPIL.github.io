import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin
    
    # Sieve primes up to 10^6
    limit_p = 10**6
    is_prime = np.ones(limit_p + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array(np.where(is_prime)[0])
    
    def compute_phi_vectorized(start, length):
        nums = np.arange(start, start + length, dtype=np.int64)
        phi_vals = nums.copy()
        rem = nums.copy()
        
        # Process primes
        # We can break if p*p > max(rem)
        # np.max(rem) is O(N). We call it in the loop.
        # To optimize, we can track max_rem, but rem changes.
        # Given N=200, np.max is very fast.
        
        for p in primes:
            p2 = p * p
            if p2 > np.max(rem):
                break
            
            mask = rem % p == 0
            if not np.any(mask):
                continue
            
            phi_vals[mask] = (phi_vals[mask] * (p - 1)) // p
            
            # Remove all factors of p
            while np.any(rem[mask] % p == 0):
                rem[mask] //= p
                
        # Handle remaining prime factors
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = (phi_vals[mask] * (rem[mask] - 1)) // rem[mask]
            
        return phi_vals

    def find_longest_distinct(vals):
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for i, v in enumerate(vals):
            if v in last_seen and last_seen[v] >= left:
                left = last_seen[v] + 1
            last_seen[v] = i
            
            if i - left + 1 > max_len:
                max_len = i - left + 1
                max_start = left
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Initial check for small s just in case
    # But random search is better.
    
    while time.time() < DEADLINE:
        s = random.randint(1, 10**12 - 200)
        phi_vals = compute_phi_vectorized(s, 200)
        k, offset = find_longest_distinct(phi_vals)
        
        if k > best_k:
            best_k = k
            best_s = s + offset
            if best_k == 200:
                break
                
        if time.time() > DEADLINE - 1.0:
            break

    return {'start': best_s, 'length': best_k}
