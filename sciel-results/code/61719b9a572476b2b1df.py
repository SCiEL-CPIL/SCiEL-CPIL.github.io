"""Rewritten program: find witnesses in every consecutive-prime gap."""
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
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s limit
    
    # The gaps are small (max 16). The primes we need to check are < gap.
    # Max gap is 16, so primes < 16 are 2, 3, 5, 7, 11, 13.
    # We can just hardcode or sieve up to 16. Sieve is safer if gaps change.
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _primes_up_to(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Primes strictly less than gap
        check_primes = [p for p in small_primes if p < gap]
        
        found = None
        # Iterate through candidates strictly between lo and hi
        for m in range(lo + 1, hi):
            is_rough = True
            for p in check_primes:
                if m % p == 0:
                    is_rough = False
                    break
            if is_rough:
                found = m
                break
        
        result[i] = found if found is not None else None

    # Ensure types are strictly Python int or None
    cleaned = []
    for v in result:
        if v is None:
            cleaned.append(None)
        else:
            # v is already a Python int from range, but int() ensures it
            cleaned.append(int(v))
            
    return cleaned
# EVOLVE-BLOCK-END
