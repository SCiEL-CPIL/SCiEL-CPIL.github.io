# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _least_prime_factor(m, small_primes):
    """Return least prime factor of m, or m if m is prime.
    small_primes is a list of primes up to sqrt(max value we might see)."""
    for p in small_primes:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Precompute primes up to sqrt(max value in gaps)
    max_val = max(hi for _, hi in P['gaps'])
    # We need primes up to sqrt(max_val) for LPF check
    # sqrt(10289) ~ 101.4, so primes up to 101 are sufficient
    # But to be safe, let's go up to 200
    small_primes = _gen_primes(200)
    
    result = [None] * len(P['gaps'])
    
    # For each gap, find a witness m strictly between lo and hi
    # such that LPF(m) >= gap_size
    # Strategy: iterate through candidates in the gap, check LPF
    # To maximize certified gaps, we should try to find witnesses for as many gaps as possible
    
    # Sort gaps by difficulty (larger gaps are easier since LPF >= gap is easier to satisfy)
    # Actually, for small gaps (2), we need LPF(m) >= 2, which means m is not divisible by 2
    # i.e., m is odd. For gap=2, lo and hi are consecutive odd primes, so there's exactly one even number between them... wait no.
    # Consecutive primes: if gap=2, lo and hi are twin primes (both odd), the number between is even.
    # LPF of an even number is 2, which equals the gap. We need STRICTLY >= gap? No, "at least the prime gap" means >= gap.
    # Wait, re-reading: "least prime factor is at least the prime gap"
    # So LPF(m) >= gap. For gap=2, LPF(m) >= 2 means m is not 1 (which it isn't) and... every integer >= 2 has LPF >= 2.
    # So for gap=2, ANY integer between lo and hi works. The only integer is lo+1 which is even, LPF=2 >= 2. ✓
    
    # For gap=4: need LPF(m) >= 4, so m must not be divisible by 2 or 3. m must be coprime to 6.
    # For larger gaps, the constraint is stronger.
    
    # Let's process gaps in order, trying to find witnesses quickly
    # Key insight: for each gap, we can check candidates from the middle outward, or just scan linearly
    # Since gaps are small (max 16), scanning is fast
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find m in (lo, hi) such that LPF(m) >= gap
        found = False
        # Try candidates in the gap
        for m in range(lo + 1, hi):
            lpf = _least_prime_factor(m, small_primes)
            if lpf >= gap:
                result[i] = int(m)
                found = True
                break
        # If not found in first pass, the gap remains None
    
    # Convert to ensure Python int types
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
