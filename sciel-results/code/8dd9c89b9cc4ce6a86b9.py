"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_fast(n):
        if n == 1:
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    def check_run(s, max_k=200):
        """Check if phi(s), ..., phi(s+max_k-1) are distinct. Returns length of distinct prefix."""
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Strategy: Try diverse starting positions near 10^12
    # Consecutive integers near 10^12: phi values are large and distinct with high probability
    # But some positions may have collisions due to shared structure
    
    # Phase 1: Check a set of strategically chosen positions near 10^12
    # Use positions that are "well-spread" to avoid correlated failures
    base = 10**12 - 200
    
    # Try offsets that are spaced out to reduce correlation
    offsets = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
               67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
               113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
               173, 179, 181, 191, 193, 197, 199]
    
    for offset in offsets:
        if time_left() < 3:
            break
        s = base - offset
        if s < 1:
            continue
        
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k == 200:
            break

    # Phase 2: If we haven't found 200, do a broader scan
    if best_k < 200 and time_left() > 5:
        # Scan a wider range with larger steps
        scan_start = base
        scan_end = max(1, base - 50000)
        
        # First try every 10th position
        for s in range(scan_start, scan_end, -10):
            if time_left() < 3:
                break
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            if best_k == 200:
                break
        
        # If still not 200, try finer scan around best found
        if best_k < 200 and time_left() > 5:
            center = best_s
            for delta in range(1, 500):
                if time_left() < 2:
                    break
                for s in [center - delta, center + delta]:
                    if s < 1 or s + 199 > 10**12:
                        continue
                    k = check_run(s, 200)
                    if k > best_k:
                        best_k = k
                        best_s = s
                        center = s
                    if best_k == 200:
                        break
                if best_k == 200:
                    break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
