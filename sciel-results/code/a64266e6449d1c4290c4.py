import math
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Fast phi function
    def phi(n):
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

    # 3. Search for a valid start s
    max_k = 200
    best_s = 1
    best_k = 1
    
    # Try large s first (less collision risk)
    candidates = []
    base = 10**11
    for i in range(500):
        candidates.append(base + i * 99991)
    # Random large values
    while len(candidates) < 5000:
        s = random.randint(10**9, 10**12 - max_k)
        candidates.append(s)
    # Add some small s as fallback
    for s in range(1, 500):
        candidates.append(s)
    
    for s in candidates:
        if time.time() > deadline:
            break
        seen = set()
        k = 0
        for i in range(max_k):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == max_k:
                break
    
    # Fallback
    if best_s + best_k - 1 > 10**12 or best_k < 1:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}
