import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def sieve_primes(limit):
        if limit < 2: return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def compute_phi_segment(L, R):
        n = R - L + 1
        rem = list(range(L, R + 1))
        phi = list(range(L, R + 1))
        
        for p in primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            for j in range(start, R + 1, p):
                idx = j - L
                if rem[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        for i in range(n):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
        return phi

    def find_best_run(phi_vals, offset):
        best_k = 1
        best_s = offset
        left = 0
        freq = {}
        for right in range(len(phi_vals)):
            val = phi_vals[right]
            freq[val] = freq.get(val, 0) + 1
            while freq[val] > 1:
                lv = phi_vals[left]
                freq[lv] -= 1
                if freq[lv] == 0:
                    del freq[lv]
                left += 1
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_s = offset + left
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Small range check
    if time_left() > 5:
        phi_small = compute_phi_segment(1, 100000)
        k, s = find_best_run(phi_small, 1)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    seg_size = 500000  # 500k
    
    while time_left() > 10:
        # Ensure we don't go out of bounds
        max_lo = 10**12 - seg_size
        if max_lo < 1:
            max_lo = 1
            
        if max_lo < 10**10:
             # If range is too small, just scan the whole thing?
             lo = 10**10
        else:
             lo = random.randint(10**10, max_lo)
        
        hi = lo + seg_size - 1
        if hi > 10**12:
            hi = 10**12
            
        if lo > hi:
            continue

        phi_seg = compute_phi_segment(lo, hi)
        k, s = find_best_run(phi_seg, lo)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # Local refinement: check around the best find
            if time_left() > 20 and best_k > 5:
                ext_lo = max(1, best_s - seg_size)
                ext_hi = min(10**12, best_s + best_k + seg_size)
                phi_ext = compute_phi_segment(ext_lo, ext_hi)
                k_ext, s_ext = find_best_run(phi_ext, ext_lo)
                if k_ext > best_k:
                    best_k = k_ext
                    best_s = s_ext

    return {'start': best_s, 'length': best_k}
