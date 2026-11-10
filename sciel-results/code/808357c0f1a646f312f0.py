import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1] using segmented sieve.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi values to the numbers themselves
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        # rem tracks the remaining part of the number to detect large prime factors
        rem = np.arange(start, start + length, dtype=np.int64)
        
        # Process small primes - only need primes where p <= start+length-1
        max_val = start + length - 1
        
        for p in primes:
            if p > max_val:
                break
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
            
            # Create indices of multiples
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            # Apply phi factor: phi(n) = n * (1 - 1/p) for each prime factor p
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            
            # Remove all factors of p from rem
            while True:
                mask = rem[indices] % p == 0
                if not np.any(mask):
                    break
                rem[indices[mask]] //= p
        
        # Handle large prime factors
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = (phi_vals[mask] // rem[mask]) * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        """
        if length <= 0:
            return 0, start
        
        phis = compute_phi_segment(start, length)
        
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            if best_k >= max_k:
                break
        
        return min(best_k, max_k), best_s

    best_k = 1
    best_s = 1
    
    # Baseline check
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 2000, 200)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    
    # Adaptive search with multiple strategies
    # Strategy 1: Large segments at structured locations
    # Strategy 2: Random sampling with larger segments
    # Strategy 3: Local refinement around found runs
    
    # First, try some structured locations with larger segments
    structured_starts = [
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
        10**6 - 5000, 10**7 - 5000, 10**8 - 5000, 10**9 - 5000,
        10**10 - 5000, 10**11 - 5000,
        2*10**6, 2*10**7, 2*10**8, 2*10**9, 2*10**10, 2*10**11,
        3*10**6, 3*10**7, 3*10**8, 3*10**9, 3*10**10, 3*10**11,
    ]
    
    seg_len = 5000  # Larger segments to find longer runs
    
    for base in structured_starts:
        if time_left() < 5:
            break
        start_det = max(1, base - seg_len // 2)
        end_det = min(10**12, start_det + seg_len)
        len_det = end_det - start_det
        if len_det < 200:
            continue
        k_det, s_det = find_best_run_sliding(start_det, len_det, 200)
        if k_det > best_k:
            best_k, best_s = k_det, s_det
            # Extend around the found run
            if time_left() > 10 and best_k > 10:
                ext_start = max(1, best_s - 1000)
                ext_end = min(10**12, best_s + best_k + 1000)
                ext_len = ext_end - ext_start
                k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
                if k_ext > best_k:
                    best_k, best_s = k_ext, s_ext
    
    # Random search with larger segments
    num_random = 0
    while time_left() > 5 and num_random < 50:
        start = random.randint(10**5, 10**12 - seg_len)
        k, s = find_best_run_sliding(start, seg_len, 200)
        if k > best_k:
            best_k, best_s = k, s
            # Extend around the found run
            if time_left() > 10 and best_k > 10:
                ext_start = max(1, best_s - 1000)
                ext_end = min(10**12, best_s + best_k + 1000)
                ext_len = ext_end - ext_start
                k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
                if k_ext > best_k:
                    best_k, best_s = k_ext, s_ext
        num_random += 1
    
    # Final refinement: try to extend the best run further
    if time_left() > 5 and best_k > 1:
        # Try extending left
        ext_start = max(1, best_s - 5000)
        ext_end = min(10**12, best_s + best_k + 5000)
        ext_len = ext_end - ext_start
        if ext_len > 200:
            k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
            if k_ext > best_k:
                best_k, best_s = k_ext, s_ext
    
    return {'start': best_s, 'length': best_k}
