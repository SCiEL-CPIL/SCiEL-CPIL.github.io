"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """Check that sum_s U[a,s]V[b,s]W[c,s] == T[a,b,c] for all a,b,c."""
    a = np.array(U, dtype=float)
    b = np.array(V, dtype=float)
    c = np.array(W, dtype=float)
    D = np.einsum('as,bs,cs->abc', a, b, c, optimize=True)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(D, T, atol=1e-6)

def construct():
    start = time.time()
    deadline = start + 115
    n, m, p = 3, 4, 8

    # Schoolbook decomposition: rank = n*m*p = 96
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Verification failed")

    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
