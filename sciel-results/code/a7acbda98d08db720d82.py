import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 100.0  # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    
    # 1. Generate primes up to 10^6 for fast phi calculation
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation with caching
    phi_cache = {}
    def phi(x):
        if x in phi_cache:
            return phi_cache[x]
        if x == 1:
            phi_cache[x] = 1
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        phi_cache[x] = result
        return result
    
    # 3. Check if a window of length k starting at s has distinct totients
    def check_window(s, k):
        seen = set()
        for i in range(s, s + k):
            p_val = phi(i)
            if p_val in seen:
                return False
            seen.add(p_val)
        return True
    
    # 4. Search Strategy: Try to find a window of length MAX_K
    # Collisions in phi(n) are rare for large n.
    # A window of 200 consecutive integers near 10^12 is very likely to have distinct totients.
    
    best_k = 1
    best_s = 1
    
    # Try random windows first
    max_start = MAX_S - MAX_K + 1
    
    # Use a fixed seed for reproducibility in case of ties
    rng = random.Random(42)
    
    # Try up to 50 random windows
    for _ in range(50):
        if time.time() - start_time > LIMIT:
            break
            
        s = rng.randint(1, max_start)
        
        if check_window(s, MAX_K):
            return {'start': s, 'length': MAX_K}
            
    # If random fails, try systematic scan from high end
    # Scan blocks of size 5000 downwards
    current_end = MAX_S
    block_size = 5000
    
    while current_end > 1:
        if time.time() - start_time > LIMIT - 10:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        # Find longest run of distinct values in this block
        seen_vals = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        r = 0
        block_len = current_end - block_start + 1
        
        for r in range(block_len):
            val = phi(block_start + r)
            if val in seen_vals and seen_vals[val] >= l:
                l = seen_vals[val] + 1
            else:
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
                if max_len >= MAX_K:
                    best_k = MAX_K
                    best_s = block_start + max_start_idx
                    return {'start': best_s, 'length': best_k}
            seen_vals[val] = r
        
        if max_len > best_k:
            best_k = max_len
            best_s = block_start + max_start_idx
            
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
