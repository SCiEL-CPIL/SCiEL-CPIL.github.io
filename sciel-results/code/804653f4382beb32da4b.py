"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Precompute small primes up to max possible gap size
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need to check for prime factors < gap_size, so primes up to max_gap-1
    small_primes = []
    for n in range(2, max_gap):
        is_prime = True
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            small_primes.append(n)
    
    result = []
    for lo, hi in P['gaps']:
        gap_size = hi - lo
        # Find primes smaller than gap_size that we need to check
        check_primes = [p for p in small_primes if p < gap_size]
        witness = None
        for m in range(lo + 1, hi):
            # Check if m has any prime factor in check_primes
            has_small_factor = False
            for p in check_primes:
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                witness = m
                break
        result.append(witness)
    
    return result
# EVOLVE-BLOCK-END
