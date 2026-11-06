"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill
    
    def now():
        return time.time() - start_time

    def sieve_primes(limit):
        """Generate list of primes up to limit using sieve of Eratosthenes."""
        if limit < 2:
            return []
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    # We need primes up to sqrt(max_n) for factorization in phi calculation.
    # If we search up to 10^7, sqrt is ~3162. If 10^8, sqrt is 10000.
    # Let's support up to 10^8 for safety, primes up to 10000.
    PRIME_LIMIT = 100000 
    primes = sieve_primes(PRIME_LIMIT)
    
    def phi_batch(start, count):
        """
        Compute phi for [start, start+count-1] using segmented sieve logic with NumPy.
        Returns a NumPy array of integers.
        """
        if count <= 0:
            return np.array([], dtype=np.int64)
        
        end = start + count - 1
        
        # Initialize phi values as the numbers themselves
        phi_vals = np.arange(start, end + 1, dtype=np.int64)
        # Remaining values to factorize
        remaining = np.arange(start, end + 1, dtype=np.int64)
        
        # Process primes
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            # We want m such that start <= m <= end and m % p == 0
            first = ((start + p - 1) // p) * p
            if first < p * p:
                first = p * p
            
            if first > end:
                continue
            
            # Get indices of multiples of p in the array
            # Indices are 0-based relative to start
            indices = np.arange(first, end + 1, p) - start
            
            # Check which of these actually have p as a factor in 'remaining'
            # This is a bit tricky because 'remaining' changes.
            # Standard segmented phi:
            # For each p, iterate multiples. If remaining[m] % p == 0, then phi[m] *= (p-1)/p and remaining[m] /= p^k.
            
            # Vectorized approach for this step:
            # We need to update phi_vals and remaining for indices where remaining[idx] % p == 0.
            
            # Get current remaining values at these indices
            rem_vals = remaining[indices]
            
            # Mask: does p divide rem_val?
            mask = (rem_vals % p == 0)
            
            if not np.any(mask):
                continue
                
            # Get the actual indices that need updating
            update_indices = indices[mask]
            
            # Update phi_vals: phi -= phi // p
            phi_vals[update_indices] -= phi_vals[update_indices] // p
            
            # Update remaining: divide by p as many times as possible
            # This is hard to vectorize perfectly without a loop per element if exponent is high,
            # but exponents are small. We can do a loop over the update_indices which is much smaller than total count.
            for idx in update_indices:
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
            
        # Handle remaining prime factors > sqrt(end)
        # If remaining[i] > 1, it is a prime factor.
        rem_mask = remaining > 1
        if np.any(rem_mask):
            rem_indices = np.where(rem_mask)[0]
            for idx in rem_indices:
                p = remaining[idx]
                phi_vals[idx] -= phi_vals[idx] // p
        
        return phi_vals

    best_k = 1
    best_s = 1
    
    # Phase 1: Exhaustive search in small range [1, 10^6]
    # This is very fast with NumPy.
    N1 = 1000000
    if now() < DEADLINE - 5:
        phi_vals = phi_batch(1, N1)
        
        # Sliding window for distinct values
        # We want max length k such that phi(s)...phi(s+k-1) are distinct.
        # Using a hash map (dict) for counts.
        freq = {}
        left = 0
        for right in range(N1):
            val = phi_vals[right]
            freq[val] = freq.get(val, 0) + 1
            while freq[val] > 1:
                left_val = phi_vals[left]
                freq[left_val] -= 1
                left += 1
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_s = left + 1 # 1-based start
        
        if now() > DEADLINE - 10:
            return {'start': best_s, 'length': best_k}

    # Phase 2: Search in larger ranges
    # Try blocks of size 100,000 in various regions
    block_size = 100000
    
    # Regions to search. We prioritize smaller numbers where runs might be longer or more common?
    # Actually, density of distinct totients might vary.
    # Let's scan a few specific ranges.
    regions = [
        (10**6, 10**7),
        (10**7, 10**8),
    ]
    
    for region_start, region_end in regions:
        if now() > DEADLINE - 10:
            break
        
        block_start = region_start
        while block_start + block_size - 1 <= region_end and now() < DEADLINE - 5:
            if block_start + block_size - 1 > 10**12:
                block_size_adj = 10**12 - block_start + 1
                if block_size_adj <= 0:
                    break
                block_size_adj = min(block_size, block_size_adj)
            else:
                block_size_adj = block_size
            
            # Compute phi for this block
            phi_block = phi_batch(block_start, block_size_adj)
            
            # Sliding window within this block
            freq = {}
            left = 0
            for right in range(block_size_adj):
                val = phi_block[right]
                freq[val] = freq.get(val, 0) + 1
                while freq[val] > 1:
                    left_val = phi_block[left]
                    freq[left_val] -= 1
                    left += 1
                k = right - left + 1
                if k > best_k:
                    best_k = k
                    best_s = block_start + left
            
            block_start += block_size

    # Phase 3: Local optimization around best found start
    if best_k > 1 and now() < DEADLINE - 5:
        for delta in range(-50, 51):
            if now() > DEADLINE - 2:
                break
            s2 = best_s + delta
            if s2 < 1 or s2 + 199 > 10**12:
                continue
            window = phi_batch(s2, 200)
            freq = {}
            left = 0
            for right in range(200):
                val = window[right]
                freq[val] = freq.get(val, 0) + 1
                while freq[val] > 1:
                    left_val = window[left]
                    freq[left_val] -= 1
                    left += 1
                k = right - left + 1
                if k > best_k:
                    best_k = k
                    best_s = s2 + left

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
