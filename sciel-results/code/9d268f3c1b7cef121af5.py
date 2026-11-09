import math
import time
import random

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
        if size <= 0:
            return []
        
        vals = [start + i for i in range(size)]
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            # Iterate over multiples of p in the segment
            for j in range(first_mult, end + 1, p):
                idx = j - start
                if rems[idx] % p == 0:
                    phis[idx] -= phis[idx] // p
                    # Remove all factors of p
                    while rems[idx] % p == 0:
                        rems[idx] //= p
                    
        # Process remaining large prime factors
        for i in range(size):
            if rems[i] > 1:
                p = rems[i]
                phis[i] -= phis[i] // p
                
        return phis

    def find_longest_run(phi_vals):
        """
        Finds the longest run of distinct values in phi_vals.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        if n == 0:
            return 0, 0
        
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
    # Use larger segments (5000) to reduce overhead
    # Start from small values and scan upward
    # Also try random positions in higher ranges
    W = 5000
    overlap = 200  # Allow runs to span segment boundaries
    
    # Phase 1: Scan from 1 upward with overlapping segments
    current_start = 1
    max_steps_phase1 = 300  # ~1.5M numbers covered
    
    for step in range(max_steps_phase1):
        if time.time() - start_time > deadline:
            break
            
        seg_start = current_start + step * (W - overlap)
        seg_end = seg_start + W - 1
        
        if seg_end > 10**12:
            seg_end = 10**12
            if seg_start > seg_end:
                break
                
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s_rel = find_longest_run(phi_vals)
            actual_s = seg_start + s_rel
            
            if k > best_k:
                best_k = k
                best_s = actual_s
                
            if best_k >= 200:
                break
                
        except Exception:
            continue

    # Phase 2: Try random positions in higher ranges if time remains
    if best_k < 200:
        random.seed(42)
        num_random_tries = 50
        for _ in range(num_random_tries):
            if time.time() - start_time > deadline:
                break
            
            # Random start between 10^6 and 10^12
            seg_start = random.randint(1000000, 10**12 - W)
            seg_end = seg_start + W - 1
            
            if seg_end > 10**12:
                seg_end = 10**12
                if seg_start > seg_end:
                    continue
                    
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
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
