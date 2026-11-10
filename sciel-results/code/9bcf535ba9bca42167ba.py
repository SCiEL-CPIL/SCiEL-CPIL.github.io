"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    # sqrt(10^12) = 10^6, so primes up to 10^6 are sufficient for factorization
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_range(start, length):
        """Compute phi(n) for n in [start, start+length-1] efficiently."""
        if length <= 0:
            return []
        
        # Initialize arrays
        # ph[i] holds the remaining part of (start+i) after dividing out small primes
        # res[i] holds the evolving phi value
        ph = [start + i for i in range(length)]
        res = [start + i for i in range(length)]
        
        # Limit for primes: we only need primes p such that p*p <= max(start+length-1)
        # However, for the sieve logic on a small range, we can just iterate all primes 
        # and check if they have multiples in the range.
        # To be safe and fast, we iterate primes up to sqrt(start+length-1).
        limit = int(math.isqrt(start + length - 1))
        
        for p in primes:
            if p > limit:
                break
            
            # Find the first multiple of p in the range [start, start+length-1]
            first_mult = ((start + p - 1) // p) * p
            if first_mult > start + length - 1:
                continue
                
            for j in range(first_mult, start + length, p):
                idx = j - start
                if ph[idx] % p == 0:
                    res[idx] -= res[idx] // p
                    while ph[idx] % p == 0:
                        ph[idx] //= p
                        
        # Handle remaining prime factors (which must be > limit, so they are prime)
        for i in range(length):
            if ph[i] > 1:
                res[i] -= res[i] // ph[i]
                
        return res

    def check_run(s, max_k=200):
        """Check if phi(s), ..., phi(s+max_k-1) are distinct. Returns length of distinct prefix."""
        vals = phi_range(s, max_k)
        seen = set()
        k = 0
        for v in vals:
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Strategy: Search for a run of length 200 near the upper limit.
    # We verify a specific block. If it fails (collision), we shift slightly.
    
    # Start near the upper limit to ensure s + k - 1 <= 10^12
    # Let's try s = 10^12 - 200
    base_s = 10**12 - 200
    
    best_k = 1
    best_s = 1

    # Try a few candidate starts near the top
    # With the faster phi_range, we can check many more candidates if needed.
    
    for offset in range(0, 100):
        if time_left() < 5:
            break
        s = base_s - offset
        if s < 1:
            continue
        
        # Check if this run has length 200
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k == 200:
            break

    # If for some reason we didn't find 200 (highly unlikely), do a small local search
    # to ensure we return a valid result, even if shorter.
    if best_k < 200 and time_left() > 10:
        # Try scanning backwards from base_s
        # We can afford to scan more due to speedup
        for s in range(base_s, base_s - 10000, -1):
            if time_left() < 5:
                break
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
