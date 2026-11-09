import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit

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
        Returns a list of phi values.
        """
        size = end - start + 1
        vals = np.arange(start, end + 1, dtype=np.int64)
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            # Create mask for positions where rems is divisible by p
            indices = np.arange(first_mult, end + 1, p)
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
        for i in range(size):
            if rems[i] > 1:
                p = rems[i]
                phis[i] -= phis[i] // p
                
        return phis.tolist()

    def find_longest_run(phi_vals, start_index):
        """
        Finds the longest run of distinct values in phi_vals.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        
        seen = set()
        current_len = 0
        current_start_rel = 0
        
        for i in range(n):
            val = phi_vals[i]
            if val in seen:
                # Collision, reset
                seen = set()
                current_start_rel = i
                current_len = 0
            seen.add(val)
            current_len += 1
            if current_len > best_k:
                best_k = current_len
                best_s_rel = current_start_rel
            if best_k >= 200:
                return best_k, best_s_rel
                
        return best_k, best_s_rel

    best_k = 1
    best_s = 1
    
    # Search strategy:
    # Try segments of different sizes at strategic locations.
    # Start with larger segments to cover more ground quickly.
    
    # First pass: scan from 10^6 with W=5000
    W = 5000
    current_start = 1000000
    max_steps = 2000  # 2000 * 5000 = 10^7 numbers
    
    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break
            
        seg_start = current_start + step * W
        seg_end = seg_start + W - 1
        
        if seg_end > 10**12:
            seg_end = 10**12
            if seg_start > seg_end:
                break
                
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s_rel = find_longest_run(phi_vals, seg_start)
            actual_s = seg_start + s_rel
            
            if k > best_k:
                best_k = k
                best_s = actual_s
                
            if best_k >= 200:
                break
                
        except Exception:
            continue
    
    # Second pass: if we haven't found k >= 100, try random spots in higher ranges
    if best_k < 100 and time.time() - start_time < deadline - 20:
        import random
        random.seed(42)
        
        # Try random starting points in [10^8, 10^12]
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            
            seg_start = random.randint(10**8, 10**12 - W)
            seg_end = seg_start + W - 1
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals, seg_start)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                if best_k >= 200:
                    break
                    
            except Exception:
                continue
    
    # Third pass: if still not good, try very small segments near 1 to find short runs
    if best_k < 50 and time.time() - start_time < deadline - 10:
        W_small = 1000
        for offset in range(0, 100000, W_small):
            if time.time() - start_time > deadline:
                break
            
            seg_start = 1 + offset
            seg_end = seg_start + W_small - 1
            
            if seg_end > 10**12:
                seg_end = 10**12
                if seg_start > seg_end:
                    break
                    
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals, seg_start)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
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
