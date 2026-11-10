"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_segmented_numpy(lo, hi):
        """Compute phi for numbers in [lo, hi] using NumPy-accelerated segmented sieve."""
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        
        if lo == 1:
            # Handle phi(1) = 1 separately
            lo = 2
            n = hi - lo + 1
            if n <= 0:
                return np.array([1], dtype=np.int64)
        
        limit = int(math.isqrt(hi)) + 1
        
        # Sieve primes up to limit using NumPy
        is_prime = np.ones(limit + 1, dtype=bool)
        if limit >= 2:
            is_prime[0] = False
            is_prime[1] = False
            for i in range(2, int(math.isqrt(limit)) + 1):
                if is_prime[i]:
                    is_prime[i*i::i] = False
            primes = np.where(is_prime)[0]
        
        # Initialize phi and remaining arrays
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            # Find indices of multiples of p in [lo, hi]
            first = ((lo + p - 1) // p) * p
            if first > hi:
                continue
            indices = np.arange(first, hi + 1, p) - lo
            
            # Apply phi adjustment for this prime
            mask = (remaining[indices] % p == 0)
            if not np.any(mask):
                continue
            
            # Update phi for these indices
            phi[indices[mask]] -= phi[indices[mask]] // p
            
            # Remove all factors of p from remaining
            for idx in indices[mask]:
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
        
        # Handle remaining prime factors
        mask = remaining > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // remaining[mask]
        
        if lo == 2:
            return phi
        else:
            return np.concatenate([np.array([1], dtype=np.int64), phi])
    
    def find_longest_distinct_run_numpy(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values using sliding window with dict."""
        n = len(phi_arr)
        if n == 0:
            return offset, 1
        
        best_k = 1
        best_s = offset
        left = 0
        count = {}
        
        for right in range(n):
            val = int(phi_arr[right])
            count[val] = count.get(val, 0) + 1
            
            while count[val] > 1:
                left_val = int(phi_arr[left])
                count[left_val] -= 1
                if count[left_val] == 0:
                    del count[left_val]
                left += 1
            
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = offset + left
            
            if best_k >= max_len:
                break
        
        return best_s, best_k
    
    best_s, best_k = 1, 1
    
    # Strategy: Search larger contiguous blocks where longer runs are more likely
    # Use multiple ranges at different scales, prioritizing larger n where phi values
    # are more likely to be distinct
    
    # Primary search ranges - larger blocks for better chance of finding long runs
    search_ranges = [
        # Very large ranges (highest priority for finding long runs)
        (999999999999 - 2000000, 999999999999),
        (900000000000 - 2000000, 900000000000),
        (800000000000 - 2000000, 800000000000),
        (700000000000 - 2000000, 700000000000),
        (600000000000 - 2000000, 600000000000),
        (500000000000 - 2000000, 500000000000),
        (400000000000 - 2000000, 400000000000),
        (300000000000 - 2000000, 300000000000),
        (200000000000 - 2000000, 200000000000),
        (100000000000 - 2000000, 100000000000),
        # Medium ranges
        (50000000000 - 1500000, 50000000000),
        (25000000000 - 1500000, 25000000000),
        (10000000000 - 1500000, 10000000000),
        (5000000000 - 1500000, 5000000000),
        (2000000000 - 1500000, 2000000000),
        # Smaller ranges as fallback
        (1000000000 - 1000000, 1000000000),
        (500000000 - 1000000, 500000000),
        (100000000 - 1000000, 100000000),
        (10000000 - 1000000, 10000000),
        (1000000 - 1000000, 1000000),
    ]
    
    for lo, hi in search_ranges:
        if time_left() < 10.0:
            break
        
        # Ensure valid range
        if lo < 1:
            lo = 1
        if hi > 10**12:
            hi = 10**12
        if lo > hi:
            continue
        
        try:
            phi_arr = compute_phi_segmented_numpy(lo, hi)
            if time_left() < 5.0:
                break
            
            s, k = find_longest_distinct_run_numpy(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
                
            # Early termination if we found a very good run
            if best_k >= 180:
                break
        except Exception:
            continue
    
    # If still not good enough, try a few more specific ranges
    if best_k < 50 and time_left() > 15.0:
        additional_ranges = [
            (1234567890123 % (10**12) - 500000, 1234567890123 % (10**12)),
            (9876543210987 % (10**12) - 500000, 9876543210987 % (10**12)),
            (123456789012 % (10**12) - 500000, 123456789012 % (10**12)),
        ]
        for lo, hi in additional_ranges:
            if time_left() < 8.0:
                break
            if lo < 1:
                lo = 1
            if hi > 10**12:
                hi = 10**12
            if lo > hi:
                continue
            try:
                phi_arr = compute_phi_segmented_numpy(lo, hi)
                s, k = find_longest_distinct_run_numpy(phi_arr, lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
