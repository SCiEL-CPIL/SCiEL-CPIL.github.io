"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute primes and their powers in 2..40
def get_prime_powers(n):
    """Return list of prime powers in [2, n], grouped by prime."""
    from math import isqrt
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for d in range(2, isqrt(num) + 1):
            if num % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    groups = {}
    for p in primes:
        powers = []
        x = p
        while x <= n:
            powers.append((x, len(powers) + 1))  # (value, exponent)
            x *= p
        groups[p] = powers
    return groups

def max_01_independent(exponents):
    """
    Given a list of positive integers (exponents), find the largest subset
    such that no element equals the sum of any non-empty subset of the others.
    For small lists, brute force.
    """
    n = len(exponents)
    best = []
    for k in range(n, 0, -1):
        for comb in combinations(exponents, k):
            comb_set = set(comb)
            valid = True
            for target in comb_set:
                # Check if target can be formed by sum of a non-empty subset of (comb_set - {target})
                others = [x for x in comb_set if x != target]
                # Subset sum check
                possible = {0}
                for x in others:
                    possible = possible | {s + x for s in possible}
                if target in possible:
                    valid = False
                    break
            if valid:
                best = list(comb)
                return best
    return best

def verify_distinct_subset_products(integers):
    """Check all 2^k subset products are distinct."""
    k = len(integers)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= integers[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    # Get prime power groups
    groups = get_prime_powers(P['n'])
    
    # For each prime, find max 0/1-independent subset of exponents
    selected = []
    for p in sorted(groups.keys()):
        powers = groups[p]
        exponents = [exp for (_, exp) in powers]
        best_exp_subset = max_01_independent(exponents)
        # Map exponents back to values
        for exp in best_exp_subset:
            val = p ** exp
            selected.append(val)
    
    # Verify
    if verify_distinct_subset_products(selected):
        return [int(x) for x in selected]
    
    # Fallback: just return the primes themselves (always valid)
    primes_only = [p for p in sorted(groups.keys())]
    return [int(x) for x in primes_only]

# EVOLVE-BLOCK-END
