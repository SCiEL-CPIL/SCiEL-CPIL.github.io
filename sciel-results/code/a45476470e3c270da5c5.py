"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

_DEADLINE = 80.0  # safety margin before 90s kill

def _prime_factorization(n):
    """Return dict of prime->exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _exponent_vector(n, primes):
    """Return tuple of exponents for each prime in primes."""
    fac = _prime_factorization(n)
    return tuple(fac.get(p, 0) for p in primes)

def _has_distinct_subset_products(nums):
    """Check if all 2^k subset products are distinct using incremental collision detection."""
    if not nums:
        return True
    products = {1}
    for x in nums:
        new_products = set()
        for p in products:
            np = p * x
            if np in products or np in new_products:
                return False
            new_products.add(np)
        products.update(new_products)
    return True

def _is_independent(vec, basis):
    """Check if vec is linearly independent from basis (list of tuples) over Q."""
    if not basis:
        return True
    from fractions import Fraction
    # Augmented matrix: basis vectors + vec
    matrix = [[Fraction(v) for v in b] for b in basis]
    matrix.append([Fraction(v) for v in vec])
    n_rows = len(matrix)
    n_cols = len(vec)
    
    rank = 0
    for col in range(n_cols):
        pivot_row = None
        for r in range(rank, n_rows):
            if matrix[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
        pivot_val = matrix[rank][col]
        for r in range(rank + 1, n_rows):
            if matrix[r][col] != 0:
                factor = matrix[r][col] / pivot_val
                for c in range(col, n_cols):
                    matrix[r][c] -= factor * matrix[rank][c]
        rank += 1
    
    # Check if last row is all zeros
    last_row = matrix[n_rows - 1]
    return not all(v == 0 for v in last_row)

def _greedy_construct(candidates, primes, deadline, rng):
    """Greedy construction with random ordering, using exponent vector linear independence."""
    best_set = []
    best_size = 0
    start_time = time.time()
    
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        current_set = []
        current_basis = []
        
        for x in shuffled:
            vec = _exponent_vector(x, primes)
            if _is_independent(vec, current_basis):
                current_set.append(x)
                current_basis.append(vec)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            if best_size >= 12:
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Primes in 2..40
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, 41) if is_prime(p)]
    all_candidates = list(range(2, 41))
    
    rng = random.Random(42)
    
    # Try greedy with random restarts
    best_set = _greedy_construct(all_candidates, primes, deadline, rng)
    
    # Verify the result
    if not _has_distinct_subset_products(best_set):
        # Fallback to primes only
        best_set = primes[:]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
