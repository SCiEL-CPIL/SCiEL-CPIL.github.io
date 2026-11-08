import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _schoolbook(n_block, m_block, p_block, i_off, j_off, k_off, m, n, p, U, V, W, cur):
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                u_idx = (i_off + i) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                U[u_idx][cur] = 1
                V[v_idx][cur] = 1
                W[w_idx][cur] = 1
                cur += 1
    return cur

def _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # M1 = (a00+a11)(b00+b11)
    U[u(0,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M2 = (a10+a11)(b00)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # M3 = (a00)(b01-b11)
    U[u(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    V[v(1,1)][cur] -= 1
    W[w(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M4 = (a11)(b10-b00)
    U[u(1,1)][cur] += 1
    V[v(1,0)][cur] += 1
    V[v(0,0)][cur] -= 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # M5 = (a00+a01)(b11)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # M6 = (a10-a00)(b00+b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M7 = (a01-a11)(b10+b11)
    U[u(0,1)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1

    return cur

def _verify(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(rank):
                    val += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Block decomposition:
    # 1. 1x4x6 (Schoolbook) -> Rank 24
    # 2. 2x2x2 (Strassen) x 6 -> Rank 42
    # Total Rank 66
    
    total_rank = 66
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    cur = 0
    # Block 1: i=0, j=0-3, k=0-5
    cur = _schoolbook(1, 4, 6, 0, 0, 0, m, n, p, U, V, W, cur)
    
    # Block 2: i=1-2, j=0-3, k=0-5
    # Split j into 0-1, 2-3
    # Split k into 0-1, 2-3, 4-5
    for m_chunk in range(2):
        j_off = m_chunk * 2
        for p_chunk in range(3):
            k_off = p_chunk * 2
            cur = _strassen_2x2x2(1, j_off, k_off, m, n, p, U, V, W, cur)
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
