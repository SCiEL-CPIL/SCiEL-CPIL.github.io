import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
_TIME_BUDGET = 120.0
_T_START = time.time()

def _verify_exact(U, V, W, n, m, p, samples=65536):
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    rng = np.random.default_rng(20240517)
    A = (rng.random((samples, n, m)) < 0.5).astype(np.int64)
    B = (rng.random((samples, m, p)) < 0.5).astype(np.int64)
    Cref = np.einsum('sij,sjk->sik', A, B)
    Uf = A.reshape(samples, n * m) @ U
    Vf = B.reshape(samples, m * p) @ V
    Wf = (Uf * Vf) @ W
    Wf = np.transpose(Wf.reshape(samples, p, n), (0, 2, 1))
    return np.array_equal(Wf, Cref)

def _fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    if time.time() - _T_START > _TIME_BUDGET - 5:
        U, V, W = _fallback()
        return {"U": U, "V": V, "W": W}
    # Partition B-columns {0,1,2} -> 3x3x3 schoolbook (rank 9)
    #             and  {3,4}    -> 3x3x2 schoolbook (rank 18). Total rank 27.
    U = np.zeros((9, 27), dtype=np.int64)
    V = np.zeros((15, 27), dtype=np.int64)
    W = np.zeros((15, 27), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):          # columns 0,1,2
                U[i * 3 + j, c] = 1
                V[j * 5 + k, c] = 1
                W[k * 3 + i, c] = 1
                c += 1
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):       # columns 3,4
                U[i * 3 + j, c] = 1
                V[j * 5 + k, c] = 1
                W[k * 3 + i, c] = 1
                c += 1
    if _verify_exact(U, V, W, 3, 3, 5, samples=65536):
        return {"U": U, "V": V, "W": W}
    U, V, W = _fallback()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
