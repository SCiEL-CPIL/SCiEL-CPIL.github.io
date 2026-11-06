"""Rewritten program: search for long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Sieve primes up to 10^6
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(1000000)
    
    def phi_single(n):
        """Compute phi(n) using precomputed primes."""
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result
    
    def compute_phi_range(start, length):
        """Compute phi for consecutive integers start, start+1, ..., start+length-1.
        Uses a segmented approach: for each prime, mark its multiples in the range."""
        if length <= 0:
            return []
        # Initialize: phi[i] = start + i
        phi_vals = list(range(start, start + length))
        
        # For each prime, update phi for multiples in range
        for p in primes:
            if p > start + length - 1 and p * p > start + length - 1:
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            if first < start:
                first = start
            # Update phi for all multiples of p in range
            for m in range(first, start + length, p):
                idx = m - start
                phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining prime factors > 10^6
        # For numbers that still have a large prime factor, we need to check
        # Actually, the segmented approach above handles all primes <= 10^6.
        # For numbers with a prime factor > 10^6, that factor must be the number itself
        # or the number divided by small factors. We need a separate pass.
        # Let's do a verification pass for large prime factors
        for i in range(length):
            n = start + i
            temp = n
            result = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    result -= result // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                result -= result // temp
            phi_vals[i] = result
        
        return phi_vals
    
    def find_best_run(phi_vals, offset, max_k=200):
        """Find the longest run of distinct phi values in phi_vals."""
        best_k = 1
        best_start = offset
        n = len(phi_vals)
        
        for s in range(n - 1):
            if time.time() > deadline:
                break
            seen = set()
            k = 0
            while s + k < n and s + k < offset + max_k and phi_vals[s + k] not in seen:
                seen.add(phi_vals[s + k])
                k += 1
            if k > best_k:
                best_k = k
                best_start = offset + s
        
        return best_k, best_start
    
    # Strategy: search multiple ranges at different scales
    best_k = 1
    best_s = 1
    
    # Phase 1: Small range baseline (quick)
    if time.time() < deadline:
        small_start = 1
        small_len = 50000
        phi_small = compute_phi_range(small_start, small_len)
        k, s = find_best_run(phi_small, small_start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 2: Medium ranges
    medium_ranges = [
        (1000000, 50000),
        (10000000, 50000),
        (100000000, 50000),
    ]
    
    for start, length in medium_ranges:
        if time.time() > deadline - 20:
            break
        if start + length - 1 > 10**12:
            continue
        phi_med = compute_phi_range(start, length)
        k, s = find_best_run(phi_med, start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 3: Large ranges - search in the 10^8 to 10^12 range
    # Longer runs are more likely here because phi values are larger and more spread out
    large_starts = [
        10**8, 5*10**8, 10**9, 5*10**9, 10**10, 5*10**10, 10**11, 5*10**11,
    ]
    
    for start in large_starts:
        if time.time() > deadline - 30:
            break
        if start > 10**12 - 200:
            continue
        length = 100000  # Search 100k consecutive numbers
        phi_large = compute_phi_range(start, length)
        k, s = find_best_run(phi_large, start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 4: Random sampling in large ranges
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        start = random.randint(10**8, 10**12 - 100000)
        length = 100000
        phi_rand = compute_phi_range(start, length)
        k, s = find_best_run(phi_rand, start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Ensure constraints
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
