import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        y = n
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

    # Deadline guard: 150s limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 140
    
    best_k = 1
    best_s = 1
    
    # Search range: large numbers where collisions are rare
    # s must be such that s + 200 - 1 <= 10^12
    min_s = 1
    max_s = 10**12 - 200
    
    # We focus on large s to minimize collision probability
    # Start from 10^11 upwards
    search_min = 10**11
    
    while time.time() < deadline:
        # Random sample s
        s = random.randint(search_min, max_s)
        
        seen = set()
        k = 0
        # Check up to 200 consecutive integers
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
