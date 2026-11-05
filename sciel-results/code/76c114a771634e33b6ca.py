import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # Leave margin for overhead
    
    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation using precomputed primes
    def phi(x):
        if x == 1:
            return 1
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    # 3. Search for a run of length 200 with distinct totients
    # We want to maximize k. Max k is 200.
    # We check if a run of length 200 exists.
    # Given the analysis, it's highly likely for large s.
    
    best_k = 1
    best_s = 1
    
    # Try a few random large starts
    # Range: 10^9 to 10^12
    random.seed(42) # For reproducibility if needed, but random is fine
    
    attempts = 0
    while time.time() - start_time < LIMIT:
        # Pick a random start s such that s + 199 <= 10^12
        s = random.randint(10**9, 10**12 - 199)
        
        seen = set()
        k = 0
        valid = True
        # Check up to 200
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                valid = False
                break
            seen.add(val)
            k += 1
            
        if valid and k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
        elif not valid and k > best_k:
            best_k = k
            best_s = s
            
        attempts += 1
        # If we've tried enough and haven't found 200, we might just return the best found
        # But 200 is very likely.
        if attempts > 100:
            break

    # Fallback: if we didn't find 200, try a structured search near 10^12?
    # Or just return what we have.
    
    return {'start': best_s, 'length': best_k}
