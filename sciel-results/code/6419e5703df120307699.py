"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    # We'll compute phi for segments and search for longest run of distinct totients
    # Strategy: use a sieve-based approach to compute phi for ranges
    
    # First, let's try small numbers with exact phi
    # For small n, we can use a simple sieve
    
    # Compute phi up to some limit using sieve
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    
    # For searching, we need to find s, k such that phi(s), ..., phi(s+k-1) are all distinct
    # and s + k - 1 <= 10^12
    
    # The key insight: for most n, phi(n) < n, and collisions happen when
    # phi(a) = phi(b) for a != b. We want to avoid such collisions in a run.
    
    # Let's use a segmented sieve approach to compute phi for large ranges.
    # For each segment [L, R], we compute phi for all numbers in that range.
    
    # To compute phi(n) for n in [L, R]:
    # 1. Find primes up to sqrt(R)
    # 2. For each prime p, for each multiple of p in [L, R], divide out p and apply (1-1/p)
    
    # Let's start with a reasonable segment size and search
    # We'll try different starting points
    
    # First, compute primes up to sqrt(10^12) = 10^6
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    if elapsed() > deadline:
        return {'start': 1, 'length': 1}
    
    # Function to compute phi for a segment [L, R]
    def compute_phi_segment(L, R):
        """Compute phi(n) for n in [L, R] using segmented sieve."""
        size = R - L + 1
        # phi_values[i] = phi(L+i), initially L+i
        phi_vals = np.arange(L, R + 1, dtype=np.int64)
        # rem[i] = remaining factor to process
        rem = phi_vals.copy()
        
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            first = ((L + p - 1) // p) * p
            if first < L:
                first = L
            for j in range(first, R + 1, p):
                idx = j - L
                if rem[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # Handle remaining prime factors
        for i in range(size):
            if rem[i] > 1:
                phi_vals[i] = phi_vals[i] // rem[i] * (rem[i] - 1)
        
        return phi_vals
    
    # Function to find longest run of distinct values in an array
    def longest_distinct_run(arr):
        """Find the longest consecutive run where all values are distinct."""
        n = len(arr)
        if n == 0:
            return 0, 0
        best_len = 1
        best_start = 0
        seen = {}
        left = 0
        for right in range(n):
            val = arr[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            if right - left + 1 > best_len:
                best_len = right - left + 1
                best_start = left
        return best_len, best_start
    
    # Search strategy:
    # 1. First try small ranges near 1
    # 2. Then try larger ranges
    
    best_k = 1
    best_s = 1
    
    # Try small range first
    small_limit = 100000
    phi_small = compute_phi_segment(1, small_limit)
    k, s = longest_distinct_run(phi_small)
    if k > best_k:
        best_k = k
        best_s = s + 1  # s is 0-indexed, we need 1-indexed
    
    if elapsed() > deadline:
        return {'start': best_s, 'length': best_k}
    
    # Now try larger segments
    # We'll search in segments of size ~500000
    seg_size = 500000
    max_n = 10**12
    
    # Try several segments at different locations
    # Focus on areas where phi values are more likely to be distinct
    # phi(n) tends to be smaller for highly composite numbers, so those cause collisions
    # We want to avoid regions with many highly composite numbers
    
    # Let's try segments starting at various points
    search_starts = [
        10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7, 10**8,
        2*10**8, 5*10**8, 10**9, 2*10**9, 5*10**9, 10**10, 2*10**10,
        5*10**10, 10**11, 2*10**11, 5*10**11
    ]
    
    for start in search_starts:
        if elapsed() > deadline - 10:
            break
        if start + seg_size > max_n:
            start = max_n - seg_size + 1
        if start < 1:
            continue
        
        phi_seg = compute_phi_segment(start, start + seg_size - 1)
        k, s_offset = longest_distinct_run(phi_seg)
        if k > best_k:
            best_k = k
            best_s = start + s_offset
    
    # Also try to extend from the best found so far
    if best_k >= 5 and elapsed() < deadline - 20:
        # Try to find a longer run by searching around the best start
        # We can try larger segments centered around promising areas
        for offset in [-seg_size, -seg_size//2, seg_size//2, seg_size]:
            new_start = best_s + offset
            if new_start < 1 or new_start + seg_size > max_n:
                continue
            if elapsed() > deadline - 10:
                break
            phi_seg = compute_phi_segment(new_start, new_start + seg_size - 1)
            k, s_offset = longest_distinct_run(phi_seg)
            if k > best_k:
                best_k = k
                best_s = new_start + s_offset
    
    # Final check: ensure result is valid
    if best_s + best_k - 1 > max_n:
        best_k = min(best_k, max_n - best_s + 1)
    if best_k < 1:
        best_s = 1
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
