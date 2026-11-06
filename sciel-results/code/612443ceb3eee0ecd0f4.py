"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import isqrt
import time

def construct():
    """
    Find the longest run of consecutive integers with distinct totients up to 10^12.
    Uses optimized segmented sieve for fast totient computation,
    then sliding window collision detection.
    """
    MAX_N = 10**12
    MAX_K = 200
    start_time = time.time()
    TIME_LIMIT = 115  # Leave 5s buffer
    
    # Precompute primes up to sqrt(MAX_N) = 10^6
    limit = isqrt(MAX_N) + 1
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Segmented phi computation: returns phi values for [start, end]
    def compute_phi_segment(start, end):
        """
        Compute phi(n) for all n in [start, end] using segmented sieve.
        Optimized for speed with vectorized operations.
        """
        size = end - start + 1
        # rem[i] will hold the remaining part of (start+i) after removing small prime factors
        rem = np.arange(start, end + 1, dtype=np.int64)
        # phi_vals starts as n, we'll apply (p-1)/p for each prime factor p
        phi_vals = np.arange(start, end + 1, dtype=np.int64)
        
        for p in small_primes:
            if p * p > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            indices = np.arange(first, end + 1, p) - start
            # Apply phi reduction once per prime factor
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            # Remove all factors of p from rem
            rem[indices] = rem[indices] // p
            # Keep removing p while it divides - use vectorized approach
            while True:
                mask = (rem[indices] % p == 0)
                if not np.any(mask):
                    break
                rem[indices[mask]] = rem[indices[mask]] // p
        
        # For each n, if rem[i] > 1, then rem[i] is a prime factor > sqrt(end)
        large_prime_mask = rem > 1
        if np.any(large_prime_mask):
            q = rem[large_prime_mask]
            phi_vals[large_prime_mask] = (phi_vals[large_prime_mask] // q) * (q - 1)
        
        # Handle n=1
        if start == 1:
            phi_vals[0] = 1
        
        return phi_vals
    
    # Find longest run of distinct phi values in an array
    def find_longest_run(phi_arr):
        """
        Find the longest contiguous subarray with all distinct values.
        Returns (best_start_offset, best_length).
        """
        n = len(phi_arr)
        if n == 0:
            return 0, 0
        best_len = 1
        best_start = 0
        
        # Use a dict to track last seen position
        last_seen = {}
        left = 0
        
        for right in range(n):
            val = phi_arr[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = left
        
        return best_start, best_len
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Scan a large contiguous range from 1
    # Use smaller segments for better cache performance and faster computation
    SEG_SIZE = 5 * 10**6
    
    # Scan from 1 to a larger range
    scan_end = min(2 * 10**7, MAX_N)
    seg_start = 1
    while seg_start <= scan_end and (time.time() - start_time) < TIME_LIMIT * 0.4:
        seg_end = min(seg_start + SEG_SIZE - 1, scan_end)
        phi_seg = compute_phi_segment(seg_start, seg_end)
        local_start, local_len = find_longest_run(phi_seg)
        
        if local_len > best_k:
            best_k = local_len
            best_s = seg_start + local_start
        
        seg_start = seg_end + 1
    
    # Phase 2: Try strategic positions where long runs might exist
    # Long runs of distinct totients are more likely in regions with
    # varied prime factorizations. Try positions that are products of
    # different small primes to get diverse phi values.
    if (time.time() - start_time) < TIME_LIMIT * 0.75:
        # Generate strategic test positions
        test_positions = []
        
        # Positions near powers of 2 (phi(2^k) = 2^(k-1), so consecutive
        # powers have distinct phis, but we need consecutive integers)
        for exp in range(20, 40):
            pos = (1 << exp) - 100  # Slightly before power of 2
            if 1 <= pos < MAX_N:
                test_positions.append(pos)
        
        # Positions that are lcm of small primes minus offset
        # This ensures diverse prime factors in the neighborhood
        lcms = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]
        for lcm_val in lcms:
            for offset in [1, 50, 100, 500, 1000]:
                pos = lcm_val - offset
                if 1 <= pos < MAX_N:
                    test_positions.append(pos)
                pos2 = lcm_val + offset
                if 1 <= pos2 < MAX_N:
                    test_positions.append(pos2)
        
        # Add some random-ish positions spread across the range
        import random
        random.seed(42)
        for _ in range(20):
            pos = random.randint(1, int(10**10))
            test_positions.append(pos)
        
        # Sort and deduplicate
        test_positions = sorted(set(test_positions))
        
        for pos in test_positions:
            if pos >= MAX_N:
                continue
            if (time.time() - start_time) > TIME_LIMIT:
                break
            seg_end = min(pos + SEG_SIZE - 1, MAX_N)
            phi_seg = compute_phi_segment(pos, seg_end)
            local_start, local_len = find_longest_run(phi_seg)
            
            if local_len > best_k:
                best_k = local_len
                best_s = pos + local_start
    
    # Phase 3: If we found a good run, try to extend it by scanning nearby
    if best_k > 10 and (time.time() - start_time) < TIME_LIMIT * 0.9:
        # Try extending the current best run by scanning a wider window around it
        extend_start = max(1, best_s - SEG_SIZE // 2)
        extend_end = min(MAX_N, best_s + best_k + SEG_SIZE // 2)
        phi_extend = compute_phi_segment(extend_start, extend_end)
        local_start, local_len = find_longest_run(phi_extend)
        
        if local_len > best_k:
            best_k = local_len
            best_s = extend_start + local_start
    
    # Cap at MAX_K
    best_k = min(best_k, MAX_K)
    
    # Verify the result
    if best_k > 1:
        # Double-check by computing phi for the specific range
        verify_end = best_s + best_k - 1
        if verify_end <= MAX_N:
            phi_verify = compute_phi_segment(best_s, verify_end)
            # Check all distinct
            seen = set()
            valid = True
            for v in phi_verify:
                if v in seen:
                    valid = False
                    break
                seen.add(v)
            if not valid:
                # Fallback: find longest valid run in this segment
                _, valid_len = find_longest_run(phi_verify)
                best_k = min(best_k, valid_len)
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
