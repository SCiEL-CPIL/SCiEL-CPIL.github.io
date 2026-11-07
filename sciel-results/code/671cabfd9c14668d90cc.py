"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _largest_prime_power(p, n):
    """Return largest power of p that is <= n."""
    result = p
    while result * p <= n:
        result *= p
    return result

def construct():
    n = 256
    primes = _sieve(n)
    
    # Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        prime_powers[p] = _largest_prime_power(p, n)
        
    # Identify swaps that improve the sum
    # A: {2, 31} -> 248
    # B: {5, 7} -> 175
    # C: {11, 17} -> 187
    # D: {13, 19} -> 247
    
    swaps = [
        ({2, 31}, 248),
        ({5, 7}, 175),
        ({11, 17}, 187),
        ({13, 19}, 247)
    ]
    
    # Build the final set
    selected = [1]
    for p in primes:
        # Check if this prime is part of a swap
        is_swapped = False
        for swap_primes, new_val in swaps:
            if p in swap_primes:
                is_swapped = True
                break
        
        if not is_swapped:
            selected.append(prime_powers[p])
            
    # Add the new values from swaps
    for swap_primes, new_val in swaps:
        selected.append(new_val)
        
    # Ensure all are Python ints and distinct
    result = [int(x) for x in selected]
    
    return result
# EVOLVE-BLOCK-END
