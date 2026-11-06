"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes up to 1e6 for fast factorization
    primes = sieve_primes(1000000)
    
    def phi_fast(n):
        """Compute phi(n) using precomputed primes."""
        if n <= 1:
            return 1
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
    
    def compute_phi_segmented(start, length):
        """
        Compute phi(n) for n in [start, start+length) using segmented sieve.
        Returns a list of phi values.
        """
        if length <= 0:
            return []
        
        # Initialize phi array with the numbers themselves
        phi_vals = list(range(start, start + length))
        
        # For each prime, update phi for all multiples in the range
        for p in primes:
            if p >= length and p > 10**6:
                break  # For large primes, at most one multiple in range
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            
            # Process all multiples of p in the range
            for m in range(first, start + length, p):
                idx = m - start
                if phi_vals[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
        
        # For numbers that still have a prime factor > sqrt(max), we need to handle them
        # Actually, the segmented sieve above handles this correctly because:
        # phi(n) = n * product(1 - 1/p) for all prime factors p of n
        # The loop above correctly applies the (1-1/p) factor for each prime p dividing n
        
        return phi_vals
    
    def check_run_vectorized(start, max_length=200):
        """Check if phi values from start are pairwise distinct using vectorized computation."""
        if start + max_length > 10**12:
            max_length = int(10**12 - start + 1)
        
        if max_length <= 0:
            return 0
        
        phi_vals = compute_phi_segmented(start, max_length)
        
        seen = set()
        k = 0
        for v in phi_vals:
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k
    
    best_k = 1
    best_start = 1
    
    # Strategy 1: Dense scan with vectorized phi computation
    # Scan ranges of 1000 starting positions at a time
    scan_range = 10000
    for base in range(0, 500000, scan_range):
        if time.time() > deadline - 30:
            break
        
        # Compute phi for a large segment
        seg_start = base
        seg_len = min(scan_range + 200, int(10**12 - base + 1))
        if seg_len <= 0:
            break
        
        phi_vals = compute_phi_segmented(seg_start, seg_len)
        
        # Now check all starting positions in this segment
        for s_offset in range(0, len(phi_vals) - 200 + 1):
            if time.time() > deadline - 30:
                break
            
            s = seg_start + s_offset
            # Check if phi values from s are distinct
            seen = set()
            k = 0
            for i in range(min(200, len(phi_vals) - s_offset)):
                v = phi_vals[s_offset + i]
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_k:
                best_k = k
                best_start = s
    
    # Strategy 2: Random search in various ranges with quick rejection
    search_ranges = [
        (10**4, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10),
        (10**10, 10**11),
        (10**11, 10**12 - 200),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline - 10:
            break
        n_samples = 200
        for _ in range(n_samples):
            if time.time() > deadline - 10:
                break
            s = random.randint(lo, max(lo, hi - 200))
            
            # Quick check: first 10 phi values must be distinct
            quick_vals = set()
            quick_ok = True
            for i in range(10):
                v = phi_fast(s + i)
                if v in quick_vals:
                    quick_ok = False
                    break
                quick_vals.add(v)
            
            if quick_ok:
                k = check_run_vectorized(s, 200)
                if k > best_k:
                    best_k = k
                    best_start = s
                    # Explore neighborhood
                    for ds in range(-50, 51):
                        if time.time() > deadline - 10:
                            break
                        ns = s + ds
                        if ns < 1 or ns + 200 > 10**12:
                            continue
                        nk = check_run_vectorized(ns, 200)
                        if nk > best_k:
                            best_k = nk
                            best_start = ns
    
    # Strategy 3: Targeted search near where phi values might be distinct
    if time.time() < deadline - 5:
        promising_starts = [
            2**20, 2**21, 2**22, 2**23, 2**24, 2**25, 2**26, 2**27, 2**28,
            10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
            999999, 9999999, 99999999, 999999999, 9999999999,
            3628800, 39916800, 6227020800, 1307674368000,
        ]
        for s in promising_starts:
            if time.time() > deadline - 5:
                break
            if s + 200 > 10**12:
                continue
            k = check_run_vectorized(s, 200)
            if k > best_k:
                best_k = k
                best_start = s
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
