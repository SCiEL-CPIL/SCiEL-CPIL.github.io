import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0 # seconds, leaving 5s margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^7 for faster phi computation
    limit_p = 10**7
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = 0
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Fast phi using precomputed primes
    def phi(x):
        if x == 1:
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
        return result

    # 3. Collision risk filter: skip starts where s or nearby numbers have high collision risk
    def is_risky(s):
        """Light filter to skip high-risk starting points"""
        if s % 4 == 0:
            return True
        if s % 6 == 0:
            return True
        return False

    # 4. Compute 200 phi values and check distinctness, tracking max consecutive distinct
    def check_run(s):
        """Returns (is_valid_for_200, max_consecutive_distinct_count)"""
        seen = set()
        for i in range(MAX_K):
            n = s + i
            p_val = phi(n)
            if p_val in seen:
                return False, i
            seen.add(p_val)
        return True, MAX_K

    # 5. Search strategy with risk filtering and best-tracking
    best_k = 1
    best_s = 1
    
    low = 10**11
    high = MAX_S - MAX_K
    rng = random.Random(42)
    
    # Phase 1: Deterministic scan of specific candidates
    # Try offsets from 10^12 down
    for offset in range(0, 200000, 50):
        if time.time() - start_time > LIMIT:
            break
        s = MAX_S - offset
        if s < low:
            break
        if is_risky(s):
            continue
        
        valid, max_distinct = check_run(s)
        if valid:
            return {'start': s, 'length': MAX_K}
        if max_distinct > best_k:
            best_k = max_distinct
            best_s = s
    
    # Phase 2: Random search with risk filter
    while time.time() - start_time < LIMIT:
        # Pick a non-risky random start
        attempts = 0
        while attempts < 50:
            s = rng.randint(low, high)
            if not is_risky(s):
                break
            attempts += 1
        else:
            continue
        
        valid, max_distinct = check_run(s)
        if valid:
            return {'start': s, 'length': MAX_K}
        if max_distinct > best_k:
            best_k = max_distinct
            best_s = s
    
    # Return best found
    if best_k >= 1:
        return {'start': best_s, 'length': best_k}
    return {'start': 1, 'length': 1}
