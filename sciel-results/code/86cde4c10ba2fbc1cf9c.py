"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes_up_to(n):
    """Return list of primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _least_prime_factor(m, primes_below_limit):
    """Return least prime factor of m, or m itself if prime.
    primes_below_limit is a list of primes to check (all < some limit)."""
    for p in primes_below_limit:
        if p * p > m:
            break
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    # Find max gap to know what primes we need
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes up to max_gap (to check if least factor >= gap)
    # Actually we need primes < max_gap to test divisibility
    primes_needed = _sieve_primes_up_to(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find primes < gap (these are the ones that would disqualify a number)
        # A number m is valid if no prime < gap divides m
        # Equivalently, least prime factor of m >= gap
        disqualifying_primes = [p for p in primes_needed if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Check if m has any prime factor < gap
            valid = True
            for p in disqualifying_primes:
                if p * p > m:
                    break
                if m % p == 0:
                    valid = False
                    break
            if valid:
                # m's least prime factor >= gap (or m is prime > gap)
                result[i] = m
                found = True
                break
    
    return result
# EVOLVE-BLOCK-END
