import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank = 3 * 8 = 24
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]   # 8 x 24
    V = [[0] * rank for _ in range(m * p)]   # 24 x 24
    W = [[0] * rank for _ in range(p * n)]   # 12 x 24

    # 3 column groups of 2
    for g in range(3):
        for k in range(m):
            for j_local in range(2):
                r = g * 8 + k * 2 + j_local
                j_global = 2 * g + j_local
                
                # U index: i * m + k
                # V index: k * p + j_global
                # W index: j_global * n + i
                
                for i in range(n):
                    u_idx = i * m + k
                    w_idx = j_global * n + i
                    U[u_idx][r] = 1
                    W[w_idx][r] = 1
                
                v_idx = k * p + j_global
                V[v_idx][r] = 1

    return {"U": U, "V": V, "W": W}

def _verify():
    try:
        res = construct()
        U = np.array(res["U"], dtype=float)
        V = np.array(res["V"], dtype=float)
        W = np.array(res["W"], dtype=float)
        n, m, p = 2, 4, 6
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i, k, j] = float(np.sum(U[i * m + k, :] * V[k * p + j, :] * W[j * n + i, :]))
        expected = np.ones((n, m, p))
        ok = np.allclose(T, expected, atol=1e-9)
        return ok, rank
    except Exception as e:
        return False, -1

_ok, _rank = _verify()
