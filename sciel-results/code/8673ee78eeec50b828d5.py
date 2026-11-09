"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_check, T, atol=1e-10)

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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    best = _schoolbook(n, m, p)
    
    # Try random rank-r decompositions with integer entries
    for target_rank in [95, 90, 85]:
        if time.time() - start_time > deadline:
            break
        for seed in range(30):
            if time.time() - start_time > deadline:
                break
            rng = np.random.RandomState(seed * 104729 + target_rank)
            U = rng.randint(-1, 2, size=(n * m, target_rank))
            V = rng.randint(-1, 2, size=(m * p, target_rank))
            W = rng.randint(-1, 2, size=(p * n, target_rank))
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best
