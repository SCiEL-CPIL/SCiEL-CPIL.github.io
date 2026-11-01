import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i * m + j, j * p + k, k * n + i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i * m + j][r] += v
        for (j, k), v in Bc.items():
            V[j * p + k][r] += v
        for (i, k), v in Cc.items():
            W[k * n + i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank, target=None):
    if target is None:
        target = _target_tensor(n, m, p)
    for (a, b, c), val in target.items():
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s - val) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _build_strassen_3x3x4(n, m, p):
    cols = []
    
    def strassen_2x2(ai, aj, bj, bk, ci, ck):
        def A(i, j): return (ai + i, aj + j)
        def B(j, k): return (bj + j, bk + k)
        def C(i, k): return (ci + i, ck + k)
        M = [
            ([(A(0,0), 1), (A(1,1), 1)], [(B(0,0), 1), (B(1,1), 1)], [(C(0,0), 1), (C(1,1), 1)]),
            ([(A(1,0), 1), (A(1,1), 1)], [(B(0,0), 1)], [(C(1,0), 1), (C(1,1), -1)]),
            ([(A(0,0), 1)], [(B(1,0), 1), (B(1,1), -1)], [(C(0,1), 1), (C(1,1), 1)]),
            ([(A(1,1), 1)], [(B(0,1), 1), (B(1,0), -1)], [(C(0,0), 1), (C(1,0), 1)]),
            ([(A(0,0), 1), (A(0,1), 1)], [(B(1,0), 1)], [(C(0,0), -1), (C(0,1), 1)]),
            ([(A(0,1), 1), (A(0,0), -1)], [(B(0,0), 1), (B(1,0), 1)], [(C(1,1), 1)]),
            ([(A(1,0), 1), (A(1,1), -1)], [(B(1,1), 1), (B(0,1), 1)], [(C(0,0), 1), (C(0,1), -1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))
    
    strassen_2x2(0, 0, 0, 0, 0, 0)      # A11 * B11 -> C11
    strassen_2x2(0, 0, 0, 2, 0, 2)      # A11 * B12 -> C12
    
    cols.append(({(0,2): 1, (1,2): 1}, {(2,0): 1, (2,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}))
    cols.append(({(0,2): 1, (1,2): 1}, {(2,2): 1, (2,3): 1}, {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}))
    
    for k in range(2):
        cols.append(({(2,0): 1, (2,1): 1}, {(0,k): 1, (1,k): 1}, {(2,k): 1}))
    for k in range(2):
        cols.append(({(2,0): 1, (2,1): 1}, {(0,2+k): 1, (1,2+k): 1}, {(2,2+k): 1}))
    cols.append(({(2,2): 1}, {(2,0): 1, (2,1): 1}, {(2,0): 1, (2,1): 1}))
    cols.append(({(2,2): 1}, {(2,2): 1, (2,3): 1}, {(2,2): 1, (2,3): 1}))
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_strassen_3x3x4(n, m, p)
    rank = len(U[0])
    
    target = _target_tensor(n, m, p)
    if not _verify(U, V, W, n, m, p, rank, target):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
