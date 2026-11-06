"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    """
    phi_val = [start + i for i in range(length)]
    rem = [start + i for i in range(length)]
    
    for p in primes:
        if p > 10**6:
            break
        if start % p == 0:
            first = start
        else:
            first = start + (p - start % p)
            
        if first > start + length - 1:
            continue
            
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """Find the longest subarray with all distinct values."""
    if not phis:
        return 0, 0
    n = len(phis)
    seen = {}
    left = 0
    max_len = 1
    max_start = 0
    for right in range(n):
        if phis[right] in seen and seen[phis[right]] >= left:
            left = seen[phis[right]] + 1
        seen[phis[right]] = right
        if right - left + 1 > max_len:
            max_len = right - left + 1
            max_start = left
    return max_len, max_start

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    random.seed(42)
    
    best_len = 1
    best_start = 1
    
    # Strategy 1: Try structured positions near primorial multiples
    # These regions have diverse prime factorizations
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    primorial = 1
    for p in small_primes[:8]:
        primorial *= p
    # primorial = 9699690
    
    for multiplier in range(100000, 100100):
        if time.time() > deadline:
            break
        s = primorial * multiplier
        if s < 1 or s + max_k - 1 > max_n:
            continue
        phis = compute_phi_segment(s, max_k, primes)
        run_len, run_start = find_longest_distinct_run(phis)
        if run_len > best_len:
            best_len = run_len
            best_start = s + run_start
        if best_len == max_k:
            return {'start': best_start, 'length': best_len}
    
    # Strategy 2: Random search
    for _ in range(2000):
        if time.time() > deadline:
            break
        s = random.randint(1, max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        run_len, run_start = find_longest_distinct_run(phis)
        if run_len > best_len:
            best_len = run_len
            best_start = s + run_start
        if best_len == max_k:
            return {'start': best_start, 'length': best_len}
    
    # Strategy 3: Try near 10^12
    for offset in range(0, 1000):
        if time.time() > deadline:
            break
        s = max_n - max_k - offset
        if s < 1:
            break
        phis = compute_phi_segment(s, max_k, primes)
        run_len, run_start = find_longest_distinct_run(phis)
        if run_len > best_len:
            best_len = run_len
            best_start = s + run_start
        if best_len == max_k:
            return {'start': best_start, 'length': best_len}
    
    return {'start': best_start, 'length': best_len}

# EVOLVE-BLOCK-END
