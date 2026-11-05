import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # seconds, leaving 5s margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    # 2. Efficient phi calculation
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

    def find_longest_run(s, max_k):
        """Find the longest valid run starting at s, up to max_k."""
        seen = set()
        for i in range(max_k):
            n = s + i
            if n > MAX_S:
                break
            p_val = phi(n)
            if p_val in seen:
                return i  # length of valid run
            seen.add(p_val)
        return max_k  # all max_k values are distinct

    best_k = 1
    best_s = 1

    # 3. Dense search: try many starting positions
    # Strategy: sample densely near 10^12 where numbers are "smooth"
    # Use a combination of dense local search and random global search
    
    # Phase 1: Dense search near 10^12
    # Check every starting position in a window
    window_start = MAX_S - 50000
    window_end = MAX_S - MAX_K
    
    # Try every 10th position first (faster coverage)
    s = window_start
    while s <= window_end:
        if time.time() - start_time > LIMIT:
            break
        k = find_longest_run(s, MAX_K)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == MAX_K:
                return {'start': best_s, 'length': best_k}
        s += 10

    # Phase 2: Finer search in promising regions
    # Re-check with step=1 in the region around best_s
    if best_k > 1:
        fine_start = max(window_start, best_s - 500)
        fine_end = min(window_end, best_s + 500)
        s = fine_start
        while s <= fine_end:
            if time.time() - start_time > LIMIT:
                break
            k = find_longest_run(s, MAX_K)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k == MAX_K:
                    return {'start': best_s, 'length': best_k}
            s += 1

    # Phase 3: Random global search in remaining time
    random.seed(42)
    while time.time() - start_time < LIMIT:
        s = random.randint(MAX_K, MAX_S - MAX_K)
        k = find_longest_run(s, MAX_K)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == MAX_K:
                return {'start': best_s, 'length': best_k}

    return {'start': best_s, 'length': best_k}
