"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061],
              [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091],
              [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111],
              [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151],
              [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177],
              [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223],
              [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259],
              [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Sieve of Eratosthenes: all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(sieve[start:n + 1:step]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    t_start = time.monotonic()
    deadline = t_start + 55.0  # safety margin before the 60 s limit

    # Max gap among the pinned pairs
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Precompute primes up to max_gap (we need primes strictly less than gap)
    all_primes = _primes_up_to(max_gap)
    
    # For each possible gap size, precompute the list of primes < that gap
    # Gaps range from 2 to max_gap
    primes_by_gap = {}
    for g in range(2, max_gap + 1):
        primes_by_gap[g] = tuple(p for p in all_primes if p < g)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.monotonic() > deadline:
            break
        
        gap = hi - lo
        check_primes = primes_by_gap.get(gap, ())
        
        # If gap <= 2, there are no primes < gap, so every integer between lo and hi works
        # But gap is at least 2 for consecutive primes (gap of 2 means twin primes, no integers between)
        # Actually for gap=2, lo+1 to hi-1 is empty range
        # For gap=3, lo+1 to hi-1 has one number, and primes < 3 is [2]
        
        found = None
        for m in range(lo + 1, hi):
            is_rough = True
            for p in check_primes:
                if m % p == 0:
                    is_rough = False
                    break
            if is_rough:
                found = m
                break
        
        result[i] = found
    
    # Ensure result is a plain list of Python ints or None
    cleaned = []
    for v in result:
        if v is None:
            cleaned.append(None)
        else:
            cleaned.append(int(v))
    return cleaned

# EVOLVE-BLOCK-END
