import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 110.0  # Leave margin for overhead
    
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

    # 3. Check if a run of length k starting at s has distinct totients
    def check_run(s, k=200):
        seen = set()
        for i in range(k):
            val = phi(s + i)
            if val in seen:
                return i  # collision at position i
            seen.add(val)
        return k  # all distinct
    
    # 4. Try deterministic large starts first
    # For large n, totients are almost certainly distinct in short runs
    test_starts = [
        10**12 - 200,
        10**12 - 400,
        999999999800,
        999999999600,
        999999999400,
        999999999200,
        999999999000,
        999999998800,
        999999998600,
        999999998400,
    ]
    
    for s in test_starts:
        if time.time() - start_time > LIMIT:
            break
        if s < 1 or s + 199 > 10**12:
            continue
        result = check_run(s, 200)
        if result == 200:
            return {'start': s, 'length': 200}
    
    # 5. Random search fallback
    random.seed(12345)
    best_k = 1
    best_s = 1
    
    attempts = 0
    while time.time() - start_time < LIMIT:
        s = random.randint(10**9, 10**12 - 199)
        result = check_run(s, 200)
        if result == 200:
            return {'start': s, 'length': 200}
        elif result > best_k:
            best_k = result
            best_s = s
        attempts += 1
        if attempts > 500:
            break
    
    return {'start': best_s, 'length': best_k}
