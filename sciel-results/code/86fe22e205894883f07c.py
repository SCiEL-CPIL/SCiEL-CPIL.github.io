import math
import time
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    # Precompute primes up to 10^6 for trial division
    LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]

    def phi(x):
        if x == 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    max_val = 10**12
    k = 200
    
    # Start at the largest possible s
    s = max_val - k + 1
    
    # We will slide the window backwards
    # Initial window: s, s+1, ..., s+k-1
    
    phi_vals = []
    counts = Counter()
    distinct_count = 0
    
    # Initialize window
    for i in range(k):
        val = phi(s + i)
        phi_vals.append(val)
        if counts[val] == 0:
            distinct_count += 1
        counts[val] += 1
        
    if distinct_count == k:
        return {'start': s, 'length': k}
        
    # Slide window
    while s > 1:
        if time.time() > DEADLINE:
            break
            
        # Remove last element (phi(s+k-1))
        last_val = phi_vals.pop()
        counts[last_val] -= 1
        if counts[last_val] == 0:
            distinct_count -= 1
            
        # Add new element (phi(s-1))
        s -= 1
        new_val = phi(s)
        phi_vals.append(new_val)
        
        if counts[new_val] == 0:
            distinct_count += 1
        counts[new_val] += 1
        
        if distinct_count == k:
            return {'start': s, 'length': k}

    # Fallback
    return {'start': 1, 'length': 14}
