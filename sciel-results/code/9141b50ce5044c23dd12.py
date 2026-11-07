"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}
_DEADLINE = 85.0  # safety margin before 90s kill

def _get_all_primes_up_to(n):
    """Sieve primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _max_exponents_for_prime(p, limit):
    """
    Find the largest set of exponents e such that p^e <= limit
    and all subset sums of the exponents are distinct.
    """
    # Get available exponents
    max_e = 0
    power = p
    while power <= limit:
        max_e += 1
        power *= p
    
    exponents = list(range(1, max_e + 1))
    
    # We want the largest subset of exponents with distinct subset sums.
    # Since max_e is small (<=5 for p=2, <=3 for p=3, <=2 for p=5),
    # we can just check all subsets.
    
    best_subset = []
    best_size = 0
    
    # Try all subsets of exponents, largest first
    for size in range(len(exponents), 0, -1):
        found = False
        for subset in combinations(exponents, size):
            # Check if subset sums are distinct
            sums = set()
            valid = True
            # Generate all subset sums
            # There are 2^size subsets
            current_sums = [0]
            for e in subset:
                new_sums = [s + e for s in current_sums]
                all_sums = current_sums + new_sums
                if len(set(all_sums)) != len(all_sums):
                    valid = False
                    break
                current_sums = all_sums
            
            if valid:
                best_subset = list(subset)
                best_size = size
                found = True
                break
        if found:
            break
    
    return best_subset

def construct():
    n = P['n']
    primes = _get_all_primes_up_to(n)
    
    result = []
    
    for p in primes:
        exponents = _max_exponents_for_prime(p, n)
        for e in exponents:
            result.append(p ** e)
            
    # Ensure we return Python ints in a list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
