"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    # Deadline guard: return best result before 85s to be safe within 90s
    deadline = time.time() + 85.0
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Determine the maximum gap size to know which small primes we need to check
    max_gap = max(hi - lo for lo, hi in gaps)
    
    # Precompute all primes up to max_gap - 1
    forbidden_primes = _sieve_primes(max_gap - 1)
    
    # Precompute forbidden prime lists for each distinct gap size
    gap_forbidden = {}
    for lo, hi in gaps:
        g = hi - lo
        if g not in gap_forbidden:
            gap_forbidden[g] = tuple(p for p in forbidden_primes if p < g)
    
    # Process each gap
    for i, (lo, hi) in enumerate(gaps):
        g = hi - lo
        forbidden = gap_forbidden[g]
        
        if not forbidden:
            # No forbidden primes: any integer in (lo, hi) works
            result[i] = lo + 1
            continue
        
        # Try candidates from lo+1 to hi-1
        for m in range(lo + 1, hi):
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                break
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure types are correct: Python int or None
    return [int(v) if v is not None else None for v in result]

# EVOLVE-BLOCK-END
