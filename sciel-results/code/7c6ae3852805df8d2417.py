import time
import numpy as np

DIMS = (2, 5, 5)
TIME_LIMIT = 135.0

def time_left():
    return TIME_LIMIT - time.time()

def get_schoolbook_decomp(n, m, p):
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}

def check_decomp(n, m, p, U, V, W):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for c in range(r):
                    acc += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if acc != 1:
                    return False
    return True

def build():
    n, m, p = DIMS
    return get_schoolbook_decomp(n, m, p)

def construct():
    try:
        res = build()
        n, m, p = DIMS
        if check_decomp(n, m, p, res['U'], res['V'], res['W']):
            return res
        else:
            return get_schoolbook_decomp(n, m, p)
    except Exception:
        n, m, p = DIMS
        return get_schoolbook_decomp(n, m, p)
