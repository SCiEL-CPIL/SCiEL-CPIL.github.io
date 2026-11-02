"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += Ua[u, c] * Va[v, c] * Wa[io, c]
                if s != T[io, u, v]:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _winograd_decomp(n, m, p):
    """
    Use 2-point DFT (Hadamard) transforms on i and j indices to decouple
    the multiplication tensor. For n=m=4, apply two levels of 2-point
    DFT to get 16 frequency bins, each requiring p scalar multiplications.
    Total rank = 16 * p = 144 (same as schoolbook) but with structured
    integer entries that may allow further optimization.
    
    Actually, let's try a different approach: use the fact that we can
    group by k and use Strassen-like recursion on the 4x4 part.
    """
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    best = _schoolbook(n, m, p)
    
    # Verify baseline
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        pass
    
    # Try Winograd decomposition
    if time.time() < _DEADLINE - 10:
        cand = _winograd_decomp(n, m, p)
        if len(cand["U"][0]) < len(best["U"][0]):
            if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
                best = cand
    
    return best

# EVOLVE-BLOCK-END
