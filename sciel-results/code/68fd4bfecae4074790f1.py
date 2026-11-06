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
        Uses a segmented approach with a 'remaining' array to handle large prime factors."""
        if length <= 0:
            return []
        
        # Initialize phi and remaining
        phi_vals = [0] * length
        remaining = [0] * length
        
        for i in range(length):
            n = start + i
            phi_vals[i] = n
            remaining[i] = n
        
        # Process small primes
        for p in primes:
            if p > start + length - 1:
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            if first < start:
                first = start
            
            # Update phi and remaining for all multiples of p in range
            for m in range(first, start + length, p):
                idx = m - start
                phi_vals[idx] -= phi_vals[idx] // p
                # Divide out all factors of p from remaining
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
        
        # Handle remaining large prime factors
        for i in range(length):
            if remaining[i] > 1:
                # remaining[i] is a prime factor
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
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
        length = 200000  # Increased length due to speedup
        phi_large = compute_phi_range(start, length)
        k, s = find_best_run(phi_large, start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 4: Random sampling in large ranges
    random.seed(42)
    for _ in range(100): # Increased number of samples
        if time.time() > deadline - 15:
            break
        start = random.randint(10**8, 10**12 - 200000)
        length = 200000
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
