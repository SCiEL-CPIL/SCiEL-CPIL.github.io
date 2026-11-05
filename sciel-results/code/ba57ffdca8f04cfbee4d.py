import math
import time
import random

# Global prime sieve
def _get_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

PRIMES = _get_primes(10**6)

def construct():
    start_time = time.time()
    LIMIT = 110.0 # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    
    # Efficient phi calculation using global primes
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in PRIMES:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    best_k = 1
    best_s = 1
    
    max_start = MAX_S - MAX_K + 1
    
    # Try random windows
    # We expect to find a valid window very quickly.
    # 100 tries is plenty.
    for _ in range(100):
        if time.time() - start_time > LIMIT:
            break
            
        s = random.randint(1, max_start)
        
        phis = []
        valid = True
        seen = set()
        # Use a list for phis if we need to backtrack, but we just break.
        for i in range(s, s + MAX_K):
            p_val = phi(i)
            if p_val in seen:
                valid = False
                break
            seen.add(p_val)
            
        if valid:
            return {'start': s, 'length': MAX_K}
            
    # Fallback: deterministic scan from high end
    # This is very unlikely to be needed.
    current_end = MAX_S
    block_size = 1000
    
    while current_end > 1:
        if time.time() - start_time > LIMIT - 5:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        phis = []
        for i in range(block_start, current_end + 1):
            phis.append(phi(i))
            
        # Find longest run of distinct values
        seen_vals = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        r = 0
        while r < len(phis):
            if phis[r] in seen_vals and seen_vals[phis[r]] >= l:
                l = seen_vals[phis[r]] + 1
            else:
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
            seen_vals[phis[r]] = r
            r += 1
            
        if max_len > best_k:
            if max_len >= MAX_K:
                best_k = MAX_K
                best_s = block_start + max_start_idx
                return {'start': best_s, 'length': best_k}
            else:
                best_k = max_len
                best_s = block_start + max_start_idx
                
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
