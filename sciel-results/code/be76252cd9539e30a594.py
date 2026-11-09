"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]
import time
import itertools
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_random_search(n, m, p, rank, max_iters, seed):
    """Random search for a rank-rank decomposition with half-integer entries."""
    rng = random.Random(seed)
    # Use small integer/half-integer values
    val_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    nm, mp, pn = n*m, m*p, p*n
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
        
        # Generate random sparse vectors
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        
        for c in range(rank):
            # Each column: pick a few non-zero entries
            for _ in range(3):  # 3 non-zeros per column
                ri = rng.randrange(nm)
                U[ri][c] = rng.choice(val_set)
            for _ in range(3):
                rj = rng.randrange(mp)
                V[rj][c] = rng.choice(val_set)
            for _ in range(3):
                rk = rng.randrange(pn)
                W[rk][c] = rng.choice(val_set)
        
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to int or half-int representation
            def to_frac(x):
                x2 = round(x * 2)
                if x2 % 2 == 0:
                    return int(x2 // 2)
                else:
                    return x2 / 2
            U = [[to_frac(v) for v in row] for row in U]
            V = [[to_frac(v) for v in row] for row in V]
            W = [[to_frac(v) for v in row] for row in W]
            return {"U": U, "V": V, "W": W}
    return None

def _structured_search(n, m, p):
    """
    Try structured low-rank decompositions.
    For 3x3 * 3x6, try rank values below 54.
    Use the fact that we need: sum_c U[i,j,c] * V[j,k,c] * W[k',i,c] = delta_{i,k'}
    where W index is k'*n + i (trace convention W[k*n+i, r]).
    """
    # Try a few structured patterns
    # Pattern: exploit the 3x3 structure
    # For each (i,k) pair where i==k, we need sum over j,c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # For i != k, sum = 0
    
    # Try rank = 27 (half of schoolbook) with structured entries
    # Actually, let's try a more systematic approach with small ranks
    # The minimum rank for 3x3 * 3x6 is at least max(9, 18, 18) = 18 by dimension arguments
    # But the actual minimum might be higher.
    
    # Let's try ranks from 18 upward with random search
    for rank in range(18, 55):
        if time.time() > _DEADLINE:
            break
        result = _try_random_search(n, m, p, rank, max_iters=2000, seed=rank*1000)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _DEADLINE
    
    # First, try structured/random search for lower rank
    result = _structured_search(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
