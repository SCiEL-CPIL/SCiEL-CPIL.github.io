import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (150s kill, 5s buffer)

    # Sieve primes up to 10^6 for factorization
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi(n) for n in [start, end] using segmented approach."""
        size = end - start + 1
        # Use int64 to avoid overflow
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        for p in primes:
            if p * p > end:
                break
            # Find indices where rem is divisible by p
            # first multiple of p >= start
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            
            # Indices in the array corresponding to multiples of p
            # start_idx is the index of the first multiple in the array
            start_idx = first - start
            # step is p
            idxs = np.arange(start_idx, size, p)
            
            if len(idxs) == 0:
                continue
            
            # Check which of these are actually divisible by p in rem
            # This handles cases where p was already divided out? 
            # No, rem is reduced by p factors in previous steps? 
            # Wait, rem is reduced by p factors for prime p. 
            # For prime q > p, rem is not divisible by p. 
            # So rem[idxs] % p == 0 is correct for identifying numbers divisible by p 
            # that haven't been fully processed? 
            # Actually, if n is divisible by p, rem[n] will be divisible by p 
            # until we divide it out. 
            # Since we process primes in increasing order, when we reach p, 
            # rem[n] still contains all factors of p. 
            # So mask is correct.
            
            mask = rem[idxs] % p == 0
            if not mask.any():
                continue
            
            sel = idxs[mask]
            
            # Update phi: phi -= phi // p
            phi[sel] -= phi[sel] // p
            
            # Update rem: divide by p as many times as possible
            rem[sel] //= p
            while True:
                still = rem[sel] % p == 0
                if not still.any():
                    break
                sel = sel[still]
                rem[sel] //= p
        
        # Handle remaining prime factors > 1
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
        
        return phi

    def find_longest_run_sliding(phi_vals):
        """Find longest consecutive run of distinct phi values using sliding window."""
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        for right in range(n):
            val = int(phi_vals[right])
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            cur_k = right - left + 1
            if cur_k > best_k:
                best_k = cur_k
                best_s = left
            if best_k >= 200:
                break
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Scan high range in large segments
    # Start from 10^12 downwards to find high-quality runs quickly
    seg_size = 5000000
    current_pos = 10**12 - seg_size + 1
    
    # We will scan downwards
    # But we must ensure s >= 1
    while current_pos >= 1:
        if time.time() - start_time > deadline - 5:
            break
        
        seg_start = current_pos
        seg_end = min(current_pos + seg_size - 1, 10**12)
        
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run_sliding(phi_vals)
            actual_s = seg_start + s
            
            if k > best_k:
                best_k = k
                best_s = actual_s
                # print(f"Found k={best_k} at s={best_s}")
                
                if best_k == 200:
                    break
        except Exception:
            pass
            
        current_pos -= seg_size

    # Phase 2: If we didn't find 200, try smaller segments in a random or structured way?
    # Or just continue scanning if time permits?
    # The loop above continues until time runs out or 200 is found.
    # If time runs out, we return the best found.

    # Final validation and clamping
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
