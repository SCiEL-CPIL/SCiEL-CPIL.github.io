"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes up to max_gap (primes < gap_size are the "bad" primes)
    small_primes = _sieve_primes(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find primes strictly less than gap
        bad_primes = [p for p in small_primes if p < gap]
        
        # Create a boolean array for the range (lo+1) to (hi-1)
        # Index 0 corresponds to lo+1, index (hi-lo-2) corresponds to hi-1
        n_candidates = hi - lo - 1
        if n_candidates <= 0:
            continue
        
        # valid[j] = True means candidate (lo+1+j) is not divisible by any bad prime
        valid = bytearray(b'\x01') * n_candidates
        
        for p in bad_primes:
            # Mark multiples of p in the range [lo+1, hi-1]
            # First multiple of p >= lo+1:
            first = ((lo + 1 + p - 1) // p) * p
            # Mark all multiples of p in [lo+1, hi-1]
            for m in range(first, hi, p):
                idx = m - (lo + 1)
                if 0 <= idx < n_candidates:
                    valid[idx] = 0
        
        # Find the first valid candidate
        for j in range(n_candidates):
            if valid[j]:
                m = lo + 1 + j
                # Verify: least prime factor of m should be >= gap
                # Quick check: m is not divisible by any prime < gap
                # Since we already marked those, this should be fine
                # But let's do a sanity check for small gaps
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
