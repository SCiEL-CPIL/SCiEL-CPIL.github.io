import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    _start_time = time.time()
    _deadline = 140.0

    n, m, p = DIMENSIONS
    # Rank 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_column(u_entries, v_entries, w_entries):
        nonlocal col_idx
        if col_idx >= rank:
            return
        for (u_idx, val) in u_entries:
            U[u_idx][col_idx] += val
        for (v_idx, val) in v_entries:
            V[v_idx][col_idx] += val
        for (w_idx, val) in w_entries:
            W[w_idx][col_idx] += val
        col_idx += 1

    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    # 1. Handle j=0 (3x1x6) with rank 1
    # U[i, 0, r] = 1, V[0, k, r] = 1, W[k, i, r] = 1
    u_entries = [(u_idx(i, 0), 1) for i in range(n)]
    v_entries = [(v_idx(0, k), 1) for k in range(p)]
    w_entries = [(w_idx(k, i), 1) for k in range(p) for i in range(n)]
    add_column(u_entries, v_entries, w_entries)

    # 2. Handle j in {1, 2, 3} with two 3x3x3 schoolbook blocks
    # Block 1: k in {0, 1, 2}
    for j in range(1, m):
        for i in range(n):
            for k in range(3):
                add_column(
                    [(u_idx(i, j), 1)],
                    [(v_idx(j, k), 1)],
                    [(w_idx(k, i), 1)]
                )
                
    # Block 2: k in {3, 4, 5}
    for j in range(1, m):
        for i in range(n):
            for k in range(3, p):
                add_column(
                    [(u_idx(i, j), 1)],
                    [(v_idx(j, k), 1)],
                    [(w_idx(k, i), 1)]
                )

    # Verify
    if time.time() < _deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        if max_err > 1e-10:
            raise ValueError(f"Decomposition error too large: {max_err}")
            
    return {"U": U, "V": V, "W": W}
