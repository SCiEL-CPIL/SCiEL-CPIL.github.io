import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # 1. Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        size = end - start + 1
        vals = np.arange(start, end + 1, dtype=np.int64)
        phi = vals.copy()
        rem = vals.copy()
        
        # Only need primes up to sqrt(end)
        max_p = int(math.isqrt(end))
        
        for p in primes:
            if p > max_p:
                break
            # Check if any number in the segment is divisible by p
            # We can optimize by only checking if p is in range? No, p is small, vals are large.
            mask = rem % p == 0
            if not mask.any():
                continue
            
            phi[mask] -= phi[mask] // p
            idxs = np.where(mask)[0]
            rem[idxs] //= p
            
            # Remove remaining factors of p
            while True:
                still_div = rem[idxs] % p == 0
                if not still_div.any():
                    break
                sub_idxs = idxs[still_div]
                rem[sub_idxs] //= p
                idxs = sub_idxs
        
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_run(phi_vals):
        best_k = 1
        best_s = 0
        n = len(phi_vals)
        
        seen = set()
        current_len = 0
        current_start = 0
        
        for i in range(n):
            val = phi_vals[i]
            if val in seen:
                seen = set()
                current_start = i
                current_len = 0
            seen.add(val)
            current_len += 1
            if current_len > best_k:
                best_k = current_len
                best_s = current_start
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Search strategy:
    # Scan segments of size 10^6 starting from 1.
    # This covers [1, 10^6], [10^6+1, 2*10^6], etc.
    # Low ranges are faster to process (fewer primes to check).
    seg_size = 1000000
    current_start = 1
    
    # We will scan as many segments as possible until deadline.
    # Prioritize lower numbers as they are faster.
    while time.time() - start_time < deadline - 5:
        seg_end = current_start + seg_size - 1
        if seg_end > 10**12:
            seg_end = 10**12
        
        try:
            phi_vals = compute_phi_segment(current_start, seg_end)
            k, s = find_longest_run(phi_vals)
            actual_s = current_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                # If we found the max possible, we can stop?
                # But maybe a longer one exists? No, max is 200.
                if best_k == 200:
                    break
        except Exception:
            pass
            
        current_start = seg_end + 1
        
        # If we've scanned up to 10^8, maybe jump to a random high number?
        # Or just continue?
        # Continuing is safer to ensure we find the best in the scanned range.
        # But if we run out of time, we might not reach high numbers.
        # Let's see how far we get.
        # 10^8 / 10^6 = 100 segments.
        # If each takes 0.5s, that's 50s. We have 110s.
        # So we can scan up to 2 * 10^8 or more.
        
    # Final safety checks
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
