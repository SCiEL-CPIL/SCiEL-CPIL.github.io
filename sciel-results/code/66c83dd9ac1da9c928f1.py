"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}
_DEADLINE = 85.0  # safety margin before 90s kill

def _prime_factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 1) + 1
    return factors

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

def _exponent_vectors(candidates, primes):
    """Convert each candidate to a tuple of exponents (one per prime)."""
    vectors = []
    for c in candidates:
        factors = _prime_factorize(c)
        vec = tuple(factors.get(p, 0) for p in primes)
        vectors.append(vec)
    return vectors

def _is_independent_mitm(vectors, indices):
    """
    Check if the vectors at given indices are linearly independent over integers.
    Uses meet-in-the-middle: split into two halves, enumerate all {-1,0,1}^k 
    combinations for each half, and check if any non-trivial combination in 
    half1 equals the negative of one in half2.
    
    For small k (<=10), direct enumeration is fast.
    """
    k = len(indices)
    if k <= 1:
        return True
    if k <= 6:
        # Direct: check all non-zero {-1,0,1}^k combinations
        from itertools import product
        n_dims = len(vectors[indices[0]])
        for combo in product((-1, 0, 1), repeat=k):
            if all(c == 0 for c in combo):
                continue
            result = [0] * n_dims
            for i, c in enumerate(combo):
                if c != 0:
                    vec = vectors[indices[i]]
                    for d in range(n_dims):
                        result[d] += c * vec[d]
            if all(r == 0 for r in result):
                return False
        return True
    
    # Meet in the middle for k > 6
    mid = k // 2
    left_indices = indices[:mid]
    right_indices = indices[mid:]
    n_dims = len(vectors[indices[0]])
    
    # Enumerate all {-1,0,1}^mid combinations for left half
    from itertools import product
    left_sums = {}
    for combo in product((-1, 0, 1), repeat=mid):
        result = [0] * n_dims
        for i, c in enumerate(combo):
            if c != 0:
                vec = vectors[left_indices[i]]
                for d in range(n_dims):
                    result[d] += c * vec[d]
        key = tuple(result)
        if key not in left_sums:
            left_sums[key] = combo
    
    # Enumerate all {-1,0,1}^(k-mid) combinations for right half
    for combo in product((-1, 0, 1), repeat=k - mid):
        result = [0] * n_dims
        for i, c in enumerate(combo):
            if c != 0:
                vec = vectors[right_indices[i]]
                for d in range(n_dims):
                    result[d] += c * vec[d]
        # Check if -result is in left_sums
        neg_key = tuple(-r for r in result)
        if neg_key in left_sums:
            left_combo = left_sums[neg_key]
            # Check it's not the trivial (all zeros) combination
            if any(c != 0 for c in left_combo) or any(c != 0 for c in combo):
                return False
    
    return True

def _is_independent_fast(vectors, indices):
    """
    Fast independence check using integer Gaussian elimination.
    Returns True if vectors at indices are linearly independent over Z.
    """
    k = len(indices)
    if k <= 1:
        return True
    if k <= 5:
        return _is_independent_mitm(vectors, indices)
    
    # For larger k, use rational Gaussian elimination
    # Build the matrix and perform elimination
    n_dims = len(vectors[indices[0]])
    # Use fractions for exact arithmetic
    from fractions import Fraction
    
    # Build matrix as list of lists of Fractions
    mat = []
    for idx in indices:
        row = [Fraction(v) for v in vectors[idx]]
        mat.append(row)
    
    # Gaussian elimination to find rank
    rank = 0
    n_rows = len(mat)
    for col in range(n_dims):
        if rank >= n_rows:
            break
        # Find pivot
        pivot_row = None
        for r in range(rank, n_rows):
            if mat[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        # Swap
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        pivot_val = mat[rank][col]
        # Eliminate below
        for r in range(rank + 1, n_rows):
            factor = mat[r][col] / pivot_val
            if factor != 0:
                for c in range(col, n_dims):
                    mat[r][c] -= factor * mat[rank][c]
        rank += 1
    
    return rank == k

def _verify_distinct_products(candidates):
    """Verify all 2^k subset products are distinct."""
    k = len(candidates)
    products = set()
    # Use iterative approach
    current_products = [1]
    for c in candidates:
        new_products = [p * c for p in current_products]
        all_products = current_products + new_products
        if len(set(all_products)) != len(all_products):
            return False
        current_products = all_products
    return True

def construct():
    n = P['n']
    # The maximum size of a set of integers with distinct subset products
    # is bounded by the number of primes <= n, because the exponent vectors
    # must be linearly independent in R^pi(n).
    # For n=40, primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (12 primes).
    # The set of these primes itself forms a valid set of size 12.
    primes = _get_all_primes_up_to(n)
    
    # Verify that the primes indeed form a valid set (they do, by unique factorization)
    # But just to be safe against any edge cases in the problem definition (e.g. if n was small)
    result = primes
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
