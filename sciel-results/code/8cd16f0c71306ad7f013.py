"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    t_start = time.time()
    deadline = t_start + 55  # 55s deadline with margin for 90s limit

    # Precompute primes up to max gap size
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    all_primes = _primes_up_to(max_gap)
    
    # For each gap, precompute the list of primes < gap
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        small_primes = [p for p in all_primes if p < gap]
        
        # Search for m in (lo, hi) with no prime factor < gap
        # Try all candidates
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            is_rough = True
            for p in small_primes:
                if m % p == 0:
                    is_rough = False
                    break
            if is_rough:
                result[i] = int(m)
                break
        
        if time.time() > deadline:
            break
    
    # Ensure result is a plain list of Python ints or None
    return [r if r is None else int(r) for r in result]

# EVOLVE-BLOCK-END
