"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import sys
from functools import lru_cache

# Precompute primes up to 256 using sieve
_MAX_N = 256

def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

_PRIMES = _sieve(_MAX_N)
_NUM_PRIMES = len(_PRIMES)

# Precompute prime factors for each number 1..256
# prime_factors[i] = list of distinct prime factors of i
_prime_factors = [[] for _ in range(_MAX_N + 1)]
for i in range(2, _MAX_N + 1):
    x = i
    factors = []
    for p in _PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            factors.append(p)
            while x % p == 0:
                x //= p
    if x > 1:
        factors.append(x)
    _prime_factors[i] = factors

# Map each prime to its index
_prime_index = {p: i for i, p in enumerate(_PRIMES)}

# For each number, compute a bitmask of its prime factors
_number_mask = [0] * (_MAX_N + 1)
for i in range(2, _MAX_N + 1):
    mask = 0
    for p in _prime_factors[i]:
        mask |= (1 << _prime_index[p])
    _number_mask[i] = mask

def construct():
    """Return a list of pairwise coprime integers from 1..256 maximizing their sum."""
    # Start with all primes + 1 as initial solution
    selected = set([1] + _PRIMES)
    
    # Local search: try to swap primes for composites to improve sum
    improved = True
    max_iterations = 100
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Try each composite number not in selected
        for num in range(2, _MAX_N + 1):
            if num in selected:
                continue
            factors = _prime_factors[num]
            if not factors:
                continue
            # Check if all prime factors of num are currently selected as primes
            # (i.e., each factor p is in selected as the number p itself)
            can_swap = all(p in selected for p in factors)
            if not can_swap:
                continue
            # Compute gain: num - sum(factors)
            gain = num - sum(factors)
            if gain > 0:
                # Perform the swap
                for p in factors:
                    selected.remove(p)
                selected.add(num)
                improved = True
    
    # Second phase: try to add composites whose prime factors are partially covered
    # by other composites (more complex swaps)
    # For now, try single swaps again in reverse order
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for num in range(_MAX_N, 1, -1):
            if num in selected:
                continue
            factors = _prime_factors[num]
            if not factors:
                continue
            # Check if we can remove some selected numbers to free up these primes
            # Simple case: all factors are in selected as primes
            can_swap = all(p in selected for p in factors)
            if not can_swap:
                continue
            gain = num - sum(factors)
            if gain > 0:
                for p in factors:
                    selected.remove(p)
                selected.add(num)
                improved = True
    
    # Convert to list of Python ints
    result = [int(x) for x in selected]
    return result

# EVOLVE-BLOCK-END
