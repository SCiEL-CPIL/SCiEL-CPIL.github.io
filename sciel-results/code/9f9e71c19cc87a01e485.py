import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s budget
    
    # Step 1: Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Compute phi values using a modified sieve approach
    # We'll compute phi for a range and scan for distinct runs
    # Strategy: compute phi for blocks, then find longest run of distinct values
    
    def compute_phi_range(start, end):
        """Compute phi(n) for n in [start, end] using trial division with precomputed primes."""
        n_values = end - start + 1
        phi_vals = list(range(start, end + 1))  # phi[n] initialized to n
        # For each prime p, for each multiple of p in [start, end], apply phi factor
        for p in primes:
            if p > end:
                break
            # First multiple of p >= start
            first = ((start + p - 1) // p) * p
            for m in range(first, end + 1, p):
                idx = m - start
                phi_vals[idx] -= phi_vals[idx] // p
        return phi_vals
    
    # We need to find the longest run of consecutive integers with distinct phi values.
    # Let's try several ranges and scan for the best run.
    
    best_k = 1
    best_s = 1
    
    # Try a large range first: [1, 500000]
    # But computing phi for 500k values with trial division by all primes up to 10^6
    # might be slow. Let's use a smarter approach.
    
    # Actually, let's compute phi for a range using the sieve method directly.
    def compute_phi_sieve(n):
        """Compute phi(1..n) using a sieve-like approach."""
        phi = list(range(n + 1))
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    # Try progressively larger ranges, keeping track of the best run found
    # First, try a moderate range to get a baseline
    test_ranges = [100000, 300000, 500000, 1000000]
    
    for max_n in test_ranges:
        if time.time() > deadline - 20:
            break
        
        # Compute phi for 1..max_n
        phi = compute_phi_sieve(max_n)
        
        # Find longest run of distinct phi values
        seen = set()
        current_k = 0
        current_start = 1
        
        for i in range(1, max_n + 1):
            if phi[i] not in seen:
                if current_k == 0:
                    current_start = i
                current_k += 1
                seen.add(phi[i])
            else:
                if current_k > best_k:
                    best_k = current_k
                    best_s = current_start
                seen = set()
                current_k = 0
        
        # Check the last run
        if current_k > best_k:
            best_k = current_k
            best_s = current_start
        
        # If we found a long run, we might want to extend the search
        if best_k >= 50:
            break
    
    # If we have time, try to search in larger ranges or different areas
    if time.time() < deadline - 30:
        # Try searching in a range further out, e.g., [10^6, 2*10^6]
        offset = 1000000
        size = 500000
        if time.time() < deadline - 20:
            phi_seg = compute_phi_range(offset, offset + size)
            seen = set()
            current_k = 0
            current_start = offset
            for i in range(size):
                if phi_seg[i] not in seen:
                    if current_k == 0:
                        current_start = offset + i
                    current_k += 1
                    seen.add(phi_seg[i])
                else:
                    if current_k > best_k:
                        best_k = current_k
                        best_s = current_start
                    seen = set()
                    current_k = 0
            if current_k > best_k:
                best_k = current_k
                best_s = current_start
    
    # Verify the result is valid
    if best_k > 200:
        best_k = 200
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    
    # Final check: verify that phi values in the range are actually distinct
    # (safety check)
    if best_k > 1 and time.time() < deadline - 5:
        def phi_single(x):
            result, d, y = x, 2, x
            while d * d <= y:
                if y % d == 0:
                    result -= result // d
                    while y % d == 0:
                        y //= d
                d += 1
            return result - result // y if y > 1 else result
        
        vals = [phi_single(best_s + i) for i in range(best_k)]
        if len(set(vals)) != best_k:
            # Fallback to a known small valid answer
            best_k = 1
            best_s = 1
    
    return {'start': best_s, 'length': best_k}
