"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = DIMENSIONS  # 2, 3, 8

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def snap(arr):
        return [[round(2*x)/2 for x in row] for row in arr]

    def try_rank(rank):
        if time.time() > DEADLINE - 10:
            return None
        for seed in range(3):
            if time.time() > DEADLINE - 5:
                return None
            np.random.seed(seed)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            for _ in range(80):
                if time.time() > DEADLINE - 5:
                    return None
                for i in range(n):
                    for j in range(m):
                        A = np.array([[V[j*p+k, r]*W[k*n+i, r] for r in range(rank)] for k in range(p)])
                        U[i*m+j] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, r]*W[k*n+i, r] for r in range(rank)] for i in range(n)])
                        V[j*p+k] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i*m+j, r]*V[j*p+k, r] for r in range(rank)] for j in range(m)])
                        W[k*n+i] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
            Us, Vs, Ws = snap(U.tolist()), snap(V.tolist()), snap(W.tolist())
            if verify(Us, Vs, Ws):
                return [Us, Vs, Ws]
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return [U.tolist(), V.tolist(), W.tolist()]
        return None

    for rank in range(12, 49):
        if time.time() > DEADLINE - 10:
            break
        res = try_rank(rank)
        if res:
            return {"U": res[0], "V": res[1], "W": res[2]}

    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

