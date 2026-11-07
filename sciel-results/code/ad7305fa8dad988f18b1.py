"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_below(n):
    """Return sorted list of primes strictly less than n."""
    if n <= 2:
        return []
    size = n
    sieve = bytearray(b'\x01') * size
    sieve[0] = 0
    sieve[1] = 0
    r = int(math.isqrt(n - 1))
    for i in range(2, r + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n:step] = bytearray(len(range(start, n, step)))
    return [i for i in range(2, n) if sieve[i]]

def _lpf_table(limit):
    """Build an array lpf where lpf[x] is the least prime factor of x (or x itself if prime), for 0<=x<=limit."""
    lpf = [0] * (limit + 1)
    if limit >= 2:
        for i in range(2, limit + 1):
            if lpf[i] == 0:
                lpf[i] = i
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        if lpf[j] == 0:
                            lpf[j] = i
    return lpf

def construct():
    t0 = time.monotonic()
    hard_deadline = t0 + 50.0

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_gap = max(hi - lo for lo, hi in gaps)
    
    # Precompute primes up to max_gap to check divisibility
    # We need primes p such that p < g.
    # Max g is 28, so primes up to 23 are sufficient.
    small_primes = _primes_below(max_gap)
    
    # Map gap size to the tuple of primes strictly less than that gap
    # This allows O(1) lookup of the forbidden divisors for a specific gap
    forbidden_by_gap = {}
    for g in set(hi - lo for lo, hi in gaps):
        forbidden_by_gap[g] = tuple(p for p in small_primes if p < g)

    for i in range(n):
        if time.monotonic() > hard_deadline:
            break
        lo, hi = gaps[i]
        g = hi - lo
        forbidden = forbidden_by_gap[g]
        
        found = False
        # Try from left
        for m in range(lo + 1, hi):
            # Check if m is divisible by any forbidden prime
            # If not, lpf(m) >= g (since any prime factor must be >= g)
            is_valid = True
            for p in forbidden:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                result[i] = m
                found = True
                break
        
        if not found:
            # Try from right
            for m in range(hi - 1, lo, -1):
                is_valid = True
                for p in forbidden:
                    if m % p == 0:
                        is_valid = False
                        break
                if is_valid:
                    result[i] = m
                    found = True
                    break

        if not found:
            result[i] = None

    # Type checking: ensure Python int
    final = []
    for v in result:
        if v is None:
            final.append(None)
        else:
            final.append(int(v))
    return final
# EVOLVE-BLOCK-END
