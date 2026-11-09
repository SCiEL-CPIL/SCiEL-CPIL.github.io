"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

# Precompute primes up to a safe limit.
# Max gap is 28, so we need primes < 28.
# But to be safe and general, let's compute up to 100.
_SMALL_PRIMES = _sieve(100)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # We need m such that least_prime_factor(m) >= gap.
        # This is equivalent to: m is not divisible by any prime q < gap.
        # Filter small primes to only those < gap
        relevant_primes = [q for q in _SMALL_PRIMES if q < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any relevant prime
            is_valid = True
            for q in relevant_primes:
                if m % q == 0:
                    is_valid = False
                    break
            
            if is_valid:
                result[i] = m
                found = True
                break
        
        # If no witness found, result[i] remains None
    
    return result
# EVOLVE-BLOCK-END
