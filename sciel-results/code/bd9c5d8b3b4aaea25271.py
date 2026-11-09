import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute prime list as numpy array for faster iteration
    primes_arr = np.array(primes, dtype=np.int64)

    def compute_phi_segment(start, end):
        """
        Computes phi(n) for n in [start, end] using segmented sieve.
        Returns a numpy array of phi values (int64).
        """
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        # Use int64 to be safe, though phi(n) < n <= 10^12 fits in int64
        vals = np.arange(start, end + 1, dtype=np.int64)
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors - use precomputed primes array
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            # Create mask for positions where rems is divisible by p
            indices = np.arange(first_mult, end + 1, p, dtype=np.int64)
            if len(indices) == 0:
                continue
            idx_array = indices - start
            
            # Only process where rems is actually divisible by p
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            # Update phi: phi[n] -= phi[n] // p
            phis[valid_idx] -= phis[valid_idx] // p
            
            # Remove all factors of p from rems
            while True:
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                valid_idx = valid_idx[still_div]
                rems[valid_idx] //= p
                    
        # Process remaining large prime factors
        # For most numbers, rems[i] will be 1 or a prime > 10^6
        # We need to handle the case where rems[i] > 1
        # Since p > 10^6, p^2 > 10^12, so rems[i] can only be 1 or a prime.
        
        # Find indices where rems > 1
        large_prime_mask = rems > 1
        if np.any(large_prime_mask):
            large_indices = np.where(large_prime_mask)[0]
            p_vals = rems[large_indices]
            phis[large_indices] -= phis[large_indices] // p_vals
            
        return phis

    def find_longest_run(phi_vals):
        """
        Finds the longest run of distinct values in phi_vals.
        phi_vals is a numpy array.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        
        if n == 0:
            return 0, 0

        # Use a dictionary to track last seen index of each value
        last_seen = {}
        current_start_rel = 0
        
        for i in range(n):
            val = phi_vals[i]
            
            if val in last_seen:
                last_idx = last_seen[val]
                if last_idx >= current_start_rel:
                    current_start_rel = last_idx + 1
            
            last_seen[val] = i
            current_len = i - current_start_rel + 1
            
            if current_len > best_k:
                best_k = current_len
                best_s_rel = current_start_rel
                
            if best_k >= 200:
                return best_k, best_s_rel
                
        return best_k, best_s_rel

    def extend_run(start, initial_k):
        """
        Try to extend a run starting at 'start' beyond initial_k.
        Returns the maximum k such that phi(start), ..., phi(start+k-1) are distinct.
        Uses incremental computation: only computes phi for the new element.
        """
        if initial_k < 1:
            return 1
        # First, compute phi for the initial range to get the set of values
        end_init = start + initial_k - 1
        if end_init > 10**12:
            end_init = 10**12
            initial_k = end_init - start + 1
        phi_vals = compute_phi_segment(start, end_init)
        seen = set(phi_vals.tolist())
        k = initial_k
        while k < 200 and start + k <= 10**12:
            # Compute phi for just the next number
            next_n = start + k
            phi_next = compute_phi_segment(next_n, next_n)
            if len(phi_next) != 1:
                break
            val = phi_next[0]
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    best_k = 1
    best_s = 1
    
    # Strategy 1: Scan contiguous regions in high ranges where totients are sparser
    # For large n, phi(n) is typically much smaller than n, reducing collision probability
    # Focus on ranges where n has many small prime factors (making phi(n) small and spread out)
    
    W = 800000  # 800k numbers per segment - larger for better coverage
    
    # Scan several segments in high ranges - focus on ranges with diverse prime factorizations
    # Numbers near multiples of primorials tend to have smaller totients
    scan_ranges = [
        (10**11, 10**11 + W * 8),
        (5 * 10**11, 5 * 10**11 + W * 8),
        (10**12 - W * 8, 10**12),
        (2 * 10**11, 2 * 10**11 + W * 6),
        (7 * 10**11, 7 * 10**11 + W * 6),
        (9 * 10**11, 9 * 10**11 + W * 6),
        (3 * 10**11, 3 * 10**11 + W * 4),
        (8 * 10**11, 8 * 10**11 + W * 4),
    ]
    
    for range_start, range_end in scan_ranges:
        if time.time() - start_time > deadline:
            break
        
        for seg_start in range(range_start, range_end, W):
            if time.time() - start_time > deadline:
                break
            
            seg_end = min(seg_start + W - 1, range_end, 10**12)
            if seg_start > seg_end:
                break
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    # Try to extend the run
                    if time.time() - start_time < deadline - 10:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
                            best_s = actual_s
                            
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}
                    
            except Exception:
                continue

    # Strategy 2: Local refinement around best found run
    # Try shifting the window to find longer runs
    if time.time() - start_time < deadline - 10 and best_k > 1:
        # Try a wider range of offsets around the best start
        for offset in range(-10, 11):
            if time.time() - start_time > deadline - 5:
                break
            trial_s = best_s + offset
            if trial_s < 1 or trial_s + 200 - 1 > 10**12:
                continue
            # Check a window of size min(200, best_k + 20) around trial_s
            window_size = min(200, best_k + 20)
            seg_end = min(trial_s + window_size - 1, 10**12)
            try:
                phi_vals = compute_phi_segment(trial_s, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = trial_s + s_rel
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    if time.time() - start_time < deadline - 10:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
                            best_s = actual_s
            except Exception:
                continue

    # Strategy 3: Additional random sampling in unexplored ranges
    if time.time() - start_time < deadline - 15:
        random.seed(42)
        
        # Sample random starting points in very high ranges
        for _ in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            # Focus on ranges where n is large but not too close to 10^12
            # Use stratified random sampling across different decades
            decade = random.choice([10**10, 10**11, 5*10**11, 9*10**11])
            seg_start = random.randint(decade, min(decade + 10**11, 10**12 - W))
            seg_end = seg_start + W - 1
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    # Try to extend
                    if time.time() - start_time < deadline - 10:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
                            best_s = actual_s
                            
                if best_k >= 200:
                    break
                    
            except Exception:
                continue

    # Strategy 4: Targeted search near numbers with special properties
    # Numbers that are products of many distinct small primes have small totients
    # Look near multiples of primorials
    if time.time() - start_time < deadline - 20:
        # Primorial 2*3*5*7*11*13*17 = 510510
        # Look near multiples of this in high ranges
        primorial = 510510
        for mult in range(2000000, 2000100):
            if time.time() - start_time > deadline - 5:
                break
            base = mult * primorial
            if base < 10**11 or base + W > 10**12:
                continue
            try:
                phi_vals = compute_phi_segment(base, base + W - 1)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = base + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    if time.time() - start_time < deadline - 10:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
                            best_s = actual_s
                            
                if best_k >= 200:
                    break
            except Exception:
                continue

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
