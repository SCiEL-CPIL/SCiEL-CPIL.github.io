import time
import math
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def now():
        return time.time()

    # --- Precompute small primes for trial division and segmented sieve ---
    LIMIT_TRIAL = 10**6
    is_prime_small = bytearray(b'\x01') * (LIMIT_TRIAL + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(math.isqrt(LIMIT_TRIAL)) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = bytearray(len(is_prime_small[i*i::i]))
    small_primes = [i for i in range(2, LIMIT_TRIAL + 1) if is_prime_small[i]]
    # Primes up to sqrt(10^12) = 10^6 are sufficient for the segmented sieve logic
    # because any number <= 10^12 has at most one prime factor > 10^6.

    def phi_segment(start, size):
        """Compute phi for [start, start+size) using segmented sieve."""
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        end = start + size
        phi = np.arange(start, end, dtype=np.int64)
        rem = np.arange(start, end, dtype=np.int64)
        
        # Process primes up to sqrt(end)
        limit = int(math.isqrt(end))
        
        for p in small_primes:
            if p > limit:
                break
            
            # Find first multiple of p in [start, end)
            first = ((start + p - 1) // p) * p
            
            # Indices in the array for multiples of p
            # Note: np.arange with step creates the indices
            indices = np.arange(first - start, size, p, dtype=np.int64)
            
            if len(indices) == 0:
                continue
            
            # Update phi: phi[n] -= phi[n] // p
            phi_vals = phi[indices]
            phi[indices] = phi_vals - phi_vals // p
            
            # Update rem: divide by p as long as divisible
            rem_vals = rem[indices].copy()
            # Loop until no more elements are divisible by p
            # This is necessary because n can be p^k
            while True:
                mask = rem_vals % p == 0
                if not np.any(mask):
                    break
                rem_vals[mask] //= p
            rem[indices] = rem_vals

        # Correction for large prime factors
        # If rem[i] > 1, it is a prime q > sqrt(end).
        # phi[i] currently is q * phi(n/q).
        # We need phi(n) = (q-1) * phi(n/q) = phi[i] - phi[i]/q.
        large_mask = rem > 1
        if np.any(large_mask):
            q = rem[large_mask]
            phi_vals = phi[large_mask]
            phi[large_mask] = phi_vals - phi_vals // q
            
        return phi

    def max_distinct_run(vals, max_k=200):
        """Find the longest run of consecutive distinct values in vals."""
        n = len(vals)
        if n == 0:
            return 0, 0
        best_k = 0
        best_idx = 0
        counts = {}
        left = 0
        
        for right in range(n):
            v = vals[right]
            counts[v] = counts.get(v, 0) + 1
            
            # Shrink window until all distinct
            while counts[v] > 1:
                v_left = vals[left]
                counts[v_left] -= 1
                if counts[v_left] == 0:
                    del counts[v_left]
                left += 1
            
            curr_k = right - left + 1
            if curr_k > best_k:
                best_k = curr_k
                best_idx = left
                if best_k >= max_k:
                    break
        return best_k, best_idx

    best_k = 1
    best_s = 1

    # --- Phase 1: Dense Scanning using Segmented Sieve ---
    # We scan contiguous blocks of size BLOCK_SIZE.
    # This allows us to cover a much larger range than the original single large array.
    
    BLOCK_SIZE = 2 * 10**6  # 2 million numbers per block
    
    # Define search regions
    # 1. Low range: 1 to 50 * 10^6
    # 2. Medium range: 10^7 to 10^7 + 50 * 10^6
    # 3. High range: 10^11 to 10^11 + 50 * 10^6
    # 4. Random high ranges if time permits
    
    search_regions = []
    
    # Low
    for i in range(25):
        s = 1 + i * BLOCK_SIZE
        search_regions.append(s)
        
    # Medium
    for i in range(25):
        s = 10**7 + i * BLOCK_SIZE
        search_regions.append(s)
        
    # High
    for i in range(25):
        s = 10**11 + i * BLOCK_SIZE
        search_regions.append(s)
        
    # Shuffle to mix low/med/high to ensure coverage if time runs out?
    # Or just sequential? Sequential is better for finding local runs if they cluster.
    # But low range is most likely to have runs?
    # Actually, runs of distinct totients are more likely to be broken by small prime collisions.
    # Large numbers have larger totients, so collisions are less frequent?
    # phi(n) ~ n. If n is large, phi(n) is large.
    # Collisions phi(a)=phi(b) are rare for large a,b.
    # So high ranges might have LONGER runs.
    # Let's prioritize high ranges.
    
    search_regions = sorted(search_regions, reverse=True)
    
    for s in search_regions:
        if now() > DEADLINE - 5.0:
            break
        if s < 1 or s + BLOCK_SIZE > 10**12:
            continue
            
        try:
            # Ensure s is int
            s_int = int(s)
            vals = phi_segment(s_int, BLOCK_SIZE)
            
            if len(vals) == 0:
                continue
                
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = s_int + idx
                if best_k == 200:
                    break
        except Exception as e:
            # If something goes wrong, skip this block
            pass

    # --- Phase 2: Fine-tune around best found ---
    if now() < DEADLINE - 3.0 and best_k < 200:
        window = 10 * BLOCK_SIZE
        s_min = max(1, best_s - window)
        s_max = min(10**12 - BLOCK_SIZE, best_s + window)
        
        # Scan with overlapping blocks
        stride = BLOCK_SIZE // 2
        
        current_s = s_min
        while current_s <= s_max:
            if now() > DEADLINE - 2.0:
                break
            if current_s < 1 or current_s + BLOCK_SIZE > 10**12:
                break
                
            try:
                vals = phi_segment(int(current_s), BLOCK_SIZE)
                if len(vals) > 0:
                    k, idx = max_distinct_run(vals, 200)
                    if k > best_k:
                        best_k = k
                        best_s = int(current_s) + idx
                        if best_k == 200:
                            break
            except Exception:
                pass
                
            current_s += stride

    return {'start': best_s, 'length': best_k}
