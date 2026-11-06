import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Precompute primes up to 10^6
    # sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def phi(n, primes):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    def max_distinct_run(s, primes, max_k):
        seen = set()
        for i in range(max_k):
            val = phi(s + i, primes)
            if val in seen:
                return i
            seen.add(val)
        return max_k

    # 2. Search for a valid window of length 200
    k_target = 200
    max_s = 10**12 - k_target + 1
    min_s = 10**11
    
    random.seed(42)
    
    best_k = 1
    best_s = 1
    
    # Main search loop: try to find k=200, track best found
    while time.time() < deadline - 2:
        s = random.randint(min_s, max_s)
        k_found = max_distinct_run(s, primes, k_target)
        
        if k_found > best_k:
            best_k = k_found
            best_s = s
            
            if best_k == k_target:
                return {'start': best_s, 'length': best_k}
                
    # If we didn't find 200, return the best we found
    # Ensure we have a valid result
    if best_k < 1:
        # Fallback to a small s if random search failed completely (unlikely)
        best_s = 1
        best_k = max_distinct_run(1, primes, 200)
        
    return {'start': best_s, 'length': best_k}
