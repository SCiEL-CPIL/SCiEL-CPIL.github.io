# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_small_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _least_prime_factor(m, primes):
    """Return least prime factor of m, or m if m is prime.
    primes must contain all primes up to sqrt(m)."""
    for p in primes:
        if p * p > m:
            break
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Precompute small primes up to sqrt(max_candidate)
    # Max candidate is ~10289, sqrt ~ 102
    # But for safety, generate up to 200
    small_primes = _sieve_small_primes(200)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m in (lo, hi) such that LPF(m) >= gap
        # LPF(m) >= gap means m has no prime factor < gap
        # So m must not be divisible by any prime < gap
        
        # Collect primes less than gap (these are the "bad" divisors)
        bad_primes = [p for p in small_primes if p < gap]
        
        # Search for m in (lo, hi) that is not divisible by any bad prime
        for m in range(lo + 1, hi):
            # Check if m is divisible by any bad prime
            valid = True
            for p in bad_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                # Verify: compute LPF and check it's >= gap
                lpf = _least_prime_factor(m, small_primes)
                if lpf >= gap:
                    result[i] = m
                    break
    
    # Ensure all entries are Python int or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result
# EVOLVE-BLOCK-END
