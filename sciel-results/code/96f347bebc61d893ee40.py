import numpy as np
import random

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _random_search(n, m, p, rank, trials=50, seed=0):
    np.random.seed(seed)
    for _ in range(trials):
        U = np.random.choice([-1., -0.5, 0., 0.5, 1.], size=(n*m, rank))
        V = np.random.choice([-1., -0.5, 0., 0.5, 1.], size=(m*p, rank))
        W = np.random.choice([-1., -0.5, 0., 0.5, 1.], size=(p*n, rank))
        if _verify(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    for rank in range(44, 39, -1):
        res = _random_search(n, m, p, rank, trials=30, seed=rank)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
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
# EVOLVE-BLOCK-END
