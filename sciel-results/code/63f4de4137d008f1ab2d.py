"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def sieve_primes(limit):
        """Sieve primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segmented(lo, hi, primes):
        """Compute phi for numbers in [lo, hi] using segmented sieve with precomputed primes."""
        n = hi - lo + 1
        if n <= 0:
            return []
        
        # phi[i] stores phi(lo+i), initialized to lo+i
        phi = [lo + i for i in range(n)]
        # remaining[i] tracks the unprocessed part
        remaining = [lo + i for i in range(n)]
        
        for p in primes:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > sqrt(hi)
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def find_best_run_sliding(phi_arr, offset, max_len=200):
        """Find longest run of distinct phi values using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0
        
        best_k = 1
        best_s = offset
        left = 0
        count = {}
        
        for right in range(n):
            val = phi_arr[right]
            count[val] = count.get(val, 0) + 1
            
            # Shrink window from left while we have duplicates
            while count[val] > 1 and left <= right:
                left_val = phi_arr[left]
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
        
        return best_s, min(best_k, max_len)
    
    best_s, best_k = 1, 1
    
    # Precompute primes up to sqrt(10^12) = 10^6
    max_prime_limit = 10**6
    primes = sieve_primes(max_prime_limit)
    
    # Strategy: Search in ranges where longer runs are likely to exist.
    # For large n, phi values are more spread out, making distinct runs more likely.
    # We'll search in multiple ranges of increasing size and location.
    
    # Phase 1: Quick scan of small ranges to establish baseline
    small_ranges = [
        (1, 5000),
        (10000, 15000),
        (100000, 105000),
        (1000000, 1005000),
    ]
    
    for lo, hi in small_ranges:
        if time_left() < 15.0:
            break
        try:
            phi_arr = compute_phi_segmented(lo, hi, primes)
            s, k = find_best_run_sliding(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 2: Larger ranges where longer runs are more likely
    # Based on theory, runs of length ~log(n)^c exist; for n~10^12, this could be ~27.6^c
    # Let's search in ranges of size ~10^6 to 10^7
    large_ranges = [
        (10**6, 10**6 + 2*10**5),
        (10**7, 10**7 + 2*10**5),
        (10**8, 10**8 + 2*10**5),
        (10**9, 10**9 + 2*10**5),
        (10**10, 10**10 + 2*10**5),
        (10**11, 10**11 + 2*10**5),
    ]
    
    for lo, hi in large_ranges:
        if time_left() < 20.0:
            break
        try:
            phi_arr = compute_phi_segmented(lo, hi, primes)
            s, k = find_best_run_sliding(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 3: If we found a good run, try to extend it
    if best_k > 15 and time_left() > 30.0:
        # Try extending around the found run
        extend_margin = 500
        extend_lo = max(1, best_s - extend_margin)
        extend_hi = min(10**12, best_s + best_k + extend_margin)
        try:
            phi_arr = compute_phi_segmented(extend_lo, extend_hi, primes)
            s, k = find_best_run_sliding(phi_arr, extend_lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            pass
    
    # Phase 4: Search in very large ranges if time permits
    if time_left() > 40.0:
        very_large_ranges = [
            (10**11, 10**11 + 10**6),
            (5*10**11, 5*10**11 + 10**6),
            (10**12 - 10**6, 10**12),
        ]
        for lo, hi in very_large_ranges:
            if time_left() < 25.0:
                break
            try:
                phi_arr = compute_phi_segmented(lo, hi, primes)
                s, k = find_best_run_sliding(phi_arr, lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
