"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook rank-n*m*p decomposition for n×m times m×p matrix multiplication.
    Uses the trace convention: output index is k*n + i for W.
    Input: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    For each term c = (i, j, k):
        U[i*m + j, c] = 1
        V[j*p + k, c] = 1
        W[k*n + i, c] = 1
    """
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    import numpy as np
    n, m, p = DIMENSIONS
    r = len(W[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    """
    Return the schoolbook rank-n*m*p decomposition for 2×5 × 5×6 matrix multiplication.
    This is the trivial upper bound; a rank-minimal exact decomposition would be better.
    """
    result = _schoolbook()
    # Verify correctness
    if not _verify(result["U"], result["V"], result["W"]):
        raise ValueError("Schoolbook decomposition failed verification")
    return result
# EVOLVE-BLOCK-END
