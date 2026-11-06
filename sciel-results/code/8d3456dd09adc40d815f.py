"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    MAX_N = 10**12
    MAX_K = 200
    
    # Precompute small primes up to 10^6 for fast factorization
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    small_primes = sieve_primes(1000000)
    
    def phi_fast(n):
        """Compute Euler's totient using precomputed small primes."""
        result = n
        y = n
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def find_run_length(start, max_len=MAX_K):
        """Find the length of consecutive distinct totients starting at 'start'."""
        seen = set()
        k = 0
        for i in range(start, min(start + max_len, MAX_N + 1)):
            if i > MAX_N:
                break
            val = phi_fast(i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    # Strategy 1: Exhaustive scan of small numbers (fast, likely to find good runs)
    # Scan up to 200,000 numbers
    scan_limit = 200000
    for s in range(1, min(scan_limit - MAX_K, MAX_N)):
        if time.time() > deadline:
            break
        k = find_run_length(s)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Strategy 2: Targeted search near primorials and special structures
    # These are known to produce longer runs of distinct totients
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Primorial-based starts
    primorials = []
    p = 2
    prod = 1
    while prod * p <= MAX_N and len(primorials) < 30:
        prod *= p
        primorials.append(prod)
        # Find next prime
        p += 1
        while not is_prime(p):
            p += 1
    
    for prim in primorials:
        for offset in range(-50, 51):
            s = prim + offset
            if s < 1 or s + MAX_K > MAX_N:
                continue
            if time.time() > deadline:
                break
            k = find_run_length(s)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Strategy 3: Random sampling across the full range
    random.seed(42)
    num_random = 5000
    for _ in range(num_random):
        if time.time() > deadline:
            break
        s = random.randint(1, MAX_N - MAX_K)
        k = find_run_length(s)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Strategy 4: Local refinement around best found
    if best_k > 1:
        for delta in range(-100, 101):
            if time.time() > deadline:
                break
            s = best_s + delta
            if s < 1 or s + MAX_K > MAX_N:
                continue
            k = find_run_length(s)
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
