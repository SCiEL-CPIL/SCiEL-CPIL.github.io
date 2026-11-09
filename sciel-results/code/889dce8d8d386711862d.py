"""Matrix multiplication 3x3 x 3x6 rank-minimal decomposition.

Attempts rank reduction from the schoolbook rank (54) via random linear
combinations of the standard basis, verified exactly with rational arithmetic.
Falls back to schoolbook if no improvement is found within the time budget.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matmul tensor.
    
    The matmul tensor T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0.
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)
    and 0 otherwise (but there are no 'otherwise' entries in the matmul tensor).
    Actually, the matmul tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    n_rows_U = len(U)
    n_rows_V = len(V)
    n_rows_W = len(W)
    if n_rows_U != n * m or n_rows_V != m * p or n_rows_W != p * n:
        return False
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _to_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is an integer)."""
    if isinstance(x, Fraction):
        return (2 * x).denominator == 1
    elif isinstance(x, (int, float)):
        return True  # int is fine; float we assume is representable
    return False

def _check_half_integer(U, V, W):
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                if not _to_half_integer(x):
                    return False
    return True

def _try_random_reduction(n, m, p, target_rank, rng, deadline):
    """Try to find a rank-reduction by randomly combining schoolbook columns."""
    base_rank = n * m * p
    U0, V0, W0 = _schoolbook(n, m, p)
    
    # Convert to Fraction for exact arithmetic
    U = [[Fraction(x) for x in row] for row in U0]
    V = [[Fraction(x) for x in row] for row in V0]
    W = [[Fraction(x) for x in row] for row in W0]
    
    # We'll try to replace some columns with random linear combinations
    # and then check if the rank can be reduced.
    # Strategy: pick a subset of columns to "merge" via random coefficients,
    # then check if the resulting tensor still decomposes correctly.
    
    # A simpler approach: try random rank-r decompositions directly.
    # For each trial, generate random U, V, W with rational entries and check.
    # This is very slow for high dimensions, so we use a smarter approach.
    
    # Better: use the fact that we can add random multiples of one column
    # to another, and if the tensor equation still holds, we haven't changed
    # the decomposition. To actually reduce rank, we need to find linear
    # dependencies in the tensor structure.
    
    # For now, let's try a direct search: for small target_rank, generate
    # random rational matrices and check.
    if target_rank >= base_rank:
        return None
    
    # This is computationally expensive, so we limit trials
    max_trials = 100
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Generate random rational entries (small integers divided by small powers of 2)
        def rand_half_int():
            num = rng.randint(-4, 4)
            den = rng.choice([1, 2])
            return Fraction(num, den)
        
        U_rand = [[rand_half_int() for _ in range(target_rank)] for _ in range(n * m)]
        V_rand = [[rand_half_int() for _ in range(target_rank)] for _ in range(m * p)]
        W_rand = [[rand_half_int() for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify_exact(U_rand, V_rand, W_rand, n, m, p):
            return U_rand, V_rand, W_rand
    
    return None

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Time budget: leave 30s margin
    deadline = time.time() + 90.0
    rng = random.Random(42)
    
    # Try to reduce rank from base_rank - 1 downward
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Only try a few target ranks to stay within time
    for target_rank in range(base_rank - 1, max(base_rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        result = _try_random_reduction(n, m, p, target_rank, rng, deadline)
        if result is not None:
            U_frac, V_frac, W_frac = result
            # Convert to float/int if possible
            U_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in U_frac]
            V_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in V_frac]
            W_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in W_frac]
            best_U, best_V, best_W = U_out, V_out, W_out
            best_rank = target_rank
            break  # Found a valid lower rank, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
