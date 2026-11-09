"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

import time
from math import isqrt

def _lpf(n):
    """Return least prime factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    d = 5
    while d * d <= n:
        if n % d == 0:
            return d
        if n % (d + 2) == 0:
            return d + 2
        d += 6
    return n

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        
        # For gap == 2: any even number in (lo, hi) works since LPF=2>=2
        if gap == 2:
            # lo and hi are both odd, so lo+1 is even and in (lo, hi)
            m = lo + 1
            result[i] = m
            continue
        
        # For larger gaps, search for m with LPF >= gap
        # Strategy: try primes first (LPF = m >= gap if m >= gap, which is always true here)
        # Then try composites with all prime factors >= gap
        
        # Try each m in range, check if LPF >= gap
        # Optimize: skip even numbers if gap > 2 (since LPF would be 2 < gap)
        # Skip multiples of small primes < gap
        
        if gap <= 2:
            m = lo + 1
            result[i] = m
            continue
        
        # For gap >= 3, m must not be divisible by any prime < gap
        # So m must be odd (not div by 2), not div by 3, 5, 7, ... up to largest prime < gap
        
        # Build list of primes < gap to use as sieve
        small_primes = []
        for p in range(2, gap):
            if _is_prime(p):
                small_primes.append(p)
        
        # Now search for m in (lo, hi) not divisible by any prime in small_primes
        # This means LPF(m) >= gap (either m is prime >= gap, or m is product of primes >= gap)
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any small prime
            valid = True
            for sp in small_primes:
                if m % sp == 0:
                    valid = False
                    break
            if valid:
                # Verify LPF >= gap
                lpf_val = _lpf(m)
                if lpf_val >= gap:
                    result[i] = m
                    found = True
                    break
        
        if not found:
            # Fallback: brute force with full LPF check
            for m in range(lo + 1, hi):
                lpf_val = _lpf(m)
                if lpf_val >= gap:
                    result[i] = m
                    found = True
                    break
    
    # Ensure all entries are Python int or None
    final = []
    for v in result:
        if v is None:
            final.append(None)
        else:
            final.append(int(v))
    return final

# EVOLVE-BLOCK-END
