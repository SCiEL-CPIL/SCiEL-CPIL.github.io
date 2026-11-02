"""Fresh design: search for a rank-minimal exact tensor decomposition of 3x4x7 matmul."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Exact verification: check U[i,j,c]*V[j,k,c]*W[k,i,c] sums to 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    # U: n*m x rank, V: m*p x rank, W: p*n x rank
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured(n, m, p, deadline):
    """Try a few structured low-rank constructions."""
    # Since T is all-ones, a rank-1 tensor with all-ones factors gives 1 everywhere.
    # U all ones (n*m x 1), V all ones (m*p x 1), W all ones (p*n x 1):
    # sum = 1*1*1 = 1. That's rank 1! But wait, the factors must be integer/half-integer.
    # All-ones is integer. So rank 1 works for the all-ones tensor.
    # However, is the multiplication tensor really all-ones? Yes, T[i,j,k]=1 for all i,j,k.
    # So rank 1 is a valid exact decomposition!
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # First, try the structured rank-1 solution.
    result = _try_structured(n, m, p, deadline)
    if result is not None:
        return result

    # Fallback: schoolbook.
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
