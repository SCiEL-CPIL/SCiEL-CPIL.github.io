"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, log2
from functools import lru_cache

def _sieve_primes(n):
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

def _factorize(x, primes):
    """Return list of exponents of x wrt primes."""
    exponents = []
    for p in primes:
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        exponents.append(e)
    return exponents

def _row_reduce(matrix):
    """Gaussian elimination over Q using exact integer arithmetic.
    Returns (rank, pivot_cols)."""
    if not matrix:
        return 0, []
    ncols = len(matrix[0])
    nrows = len(matrix)
    # Work on a copy as list of lists
    mat = [row[:] for row in matrix]
    rank = 0
    pivot_cols = []
    for col in range(ncols):
        # Find pivot
        pivot_row = -1
        for r in range(rank, nrows):
            if mat[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        # Swap
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        pivot_val = mat[rank][col]
        # Eliminate below
        for r in range(rank + 1, nrows):
            factor = mat[r][col]
            if factor != 0:
                # mat[r] = mat[r] * pivot_val - factor * mat[rank]
                for c in range(col, ncols):
                    mat[r][c] = mat[r][c] * pivot_val - factor * mat[rank][c]
                # Scale to keep numbers manageable (optional: divide by gcd)
                # For correctness we don't need to reduce, but let's try
                from math import gcd
                g = 0
                for c in range(col, ncols):
                    g = gcd(g, abs(mat[r][c]))
                if g > 1:
                    for c in range(col, ncols):
                        mat[r][c] //= g
        rank += 1
        pivot_cols.append(col)
        if rank == nrows:
            break
    return rank, pivot_cols

def _check_independent(vectors):
    """Check if a list of integer vectors are linearly independent over Q."""
    if not vectors:
        return True
    # Build matrix: rows = vectors
    rank, _ = _row_reduce(vectors)
    return rank == len(vectors)

def _construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin within 60s (killed at 90s)
    n = 40
    primes = _sieve_primes(n)
    
    # Factorize all numbers 2..40
    all_vecs = {}
    for x in range(2, n + 1):
        all_vecs[x] = _factorize(x, primes)
    
    # Precompute: for each number, its vector
    # We want to find the largest subset with independent vectors
    
    # Upper bound: the dimension of the space is len(primes) = 12
    # So max possible size is 12. But we need to verify.
    # Actually, the vectors are in R^12 (12 primes up to 40: 2,3,5,7,11,13,17,19,23,29,31,37)
    # So max independent set has size at most 12.
    
    # Let's verify: can we find 12 independent vectors?
    # The prime numbers themselves give 12 independent unit vectors.
    # So 12 is achievable. Can we do better? No, because we're in a 12-dimensional space.
    
    # Wait - let me recount primes up to 40:
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 -> 12 primes
    # So the maximum possible is 12.
    
    # Let's verify that all 12 primes work:
    prime_set = list(range(2, 41))
    prime_nums = [x for x in prime_set if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
    
    # Verify all 12 primes are independent
    prime_vecs = [all_vecs[p] for p in prime_nums]
    assert _check_independent(prime_vecs), "Primes should be independent"
    
    # Since the ambient space is 12-dimensional, we cannot exceed 12.
    # The 12 primes give us exactly 12.
    
    # But wait - let me double check the constraint. The task says "1..18 distinct integers in 2..40".
    # The upper bound of 18 is just the contract's maximum, not a mathematical bound.
    # The mathematical bound is the dimension of the prime exponent space = 12.
    
    # So the answer is 12, achieved by the 12 primes.
    
    # Let's verify this is correct by checking that no 13-element subset works.
    # Since we're in a 12-dim space, any 13 vectors are linearly dependent,
    # so no 13-element subset can have all distinct subset products.
    
    result = list(prime_nums)
    # Verify
    assert len(result) == 12
    assert all(isinstance(x, int) for x in result)
    assert all(2 <= x <= 40 for x in result)
    assert len(set(result)) == 12
    
    return result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
