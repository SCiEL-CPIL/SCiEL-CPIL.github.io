"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Precompute primes up to the maximum gap size
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # Sieve of Eratosthenes
    is_prime = [True] * (max_gap + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_gap ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_gap + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m such that LPF(m) >= gap
        # This is equivalent to m not being divisible by any prime p < gap
        # Get relevant primes
        relevant_primes = [p for p in primes if p < gap]
        
        for m in range(lo + 1, hi):
            valid = True
            for p in relevant_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                break
    return result
# EVOLVE-BLOCK-END
