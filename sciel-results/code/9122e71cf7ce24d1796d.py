import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

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
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            if first_mult > end:
                continue
                
            # Use direct index computation instead of np.arange
            # Number of multiples: floor((end - first_mult) / p) + 1
            count = (end - first_mult) // p + 1
            if count <= 0:
                continue
            
            # Indices of multiples of p in [start, end]
            idx_array = np.arange(first_mult - start, size, p, dtype=np.int64)
            
            # Only process where rems is actually divisible by p
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            # Update phi: phi[n] -= phi[n] // p
            phis[valid_idx] -= phis[valid_idx] // p
            
            # Remove all factors of p from rems - do it in one pass
            # For each valid_idx, divide rems by p as many times as possible
            # Use a loop but with early exit
            for _ in range(10):  # p^10 > 10^12 for p >= 2
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                rems[valid_idx[still_div]] //= p
                    
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

    def find_longest_run(phi_vals, start_index):
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
        current_len = 0
        current_start_rel = 0
        
        for i in range(n):
            val = int(phi_vals[i])
            
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

    def check_run_distinct(start, k):
        """
        Check if phi(start), phi(start+1), ..., phi(start+k-1) are all distinct.
        Returns True if distinct, False otherwise.
        """
        if start < 1 or start + k - 1 > 10**12:
            return False
        if k <= 1:
            return True
        
        # Compute phi for the range
        phi_vals = compute_phi_segment(start, start + k - 1)
        if len(phi_vals) != k:
            return False
        
        # Check for duplicates
        seen = set()
        for val in phi_vals:
            v = int(val)
            if v in seen:
                return False
            seen.add(v)
        return True

    def extend_run(start, initial_k):
        """
        Try to extend a run starting at 'start' beyond initial_k.
        Returns the maximum k such that phi(start), ..., phi(start+k-1) are distinct.
        """
        k = initial_k
        while k < 200 and start + k <= 10**12:
            if check_run_distinct(start, k + 1):
                k += 1
            else:
                break
        return k

    best_k = 1
    best_s = 1
    
    # Strategy 1: Scan contiguous regions in high ranges where totients are sparser
    # For large n, phi(n) is typically much smaller than n, reducing collision probability
    
    W = 200000  # 200k numbers per segment - smaller for more segments
    
    # Scan several segments in high ranges - focus on ranges likely to have long runs
    scan_ranges = [
        (8 * 10**11, 8 * 10**11 + W * 10),
        (9 * 10**11, 9 * 10**11 + W * 10),
        (10**12 - W * 10, 10**12),
        (5 * 10**11, 5 * 10**11 + W * 5),
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
                k, s_rel = find_longest_run(phi_vals, seg_start)
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

    # Strategy 2: Random sampling in high ranges
    if time.time() - start_time < deadline - 15:
        random.seed(42)
        
        # Sample random starting points in very high ranges
        for _ in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            # Focus on ranges where n is large
            seg_start = random.randint(7 * 10**11, 10**12 - W)
            seg_end = seg_start + W - 1
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals, seg_start)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    # Try to extend
                    if time.time() - start_time < deadline - 5:
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
