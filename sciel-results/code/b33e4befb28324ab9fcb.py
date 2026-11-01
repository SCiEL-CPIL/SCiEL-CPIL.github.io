import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, atol=1e-10):
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > atol:
                    return False
    return True

def _build_block_232():
    """
    Rank-11 decomposition for 2x3 * 3x2 matmul.
    U: (6, 11), V: (6, 11), W: (4, 11)
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # m0 = (a00+a11)(b00+b11)
    U[0][0] = 1.0; U[4][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0

    # m1 = (a10+a11)b00
    U[3][1] = 1.0; U[4][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0

    # m2 = a00(b10-b11)
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0

    # m3 = a11(b01-b00)
    U[4][3] = 1.0
    V[1][3] = 1.0; V[0][3] = -1.0
    W[0][3] = -1.0; W[2][3] = 1.0

    # m4 = (a00+a01)b11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[1][4] = 1.0; W[3][4] = 1.0

    # m5 = (a10-a00)(b00+b01)
    U[3][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[0][5] = 1.0

    # m6 = (a01-a11)(b10+b11)
    U[1][6] = 1.0; U[4][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0

    # Schoolbook for j=2
    # a02 * b20 -> C00
    U[2][7] = 1.0
    V[4][7] = 1.0
    W[0][7] = 1.0

    # a02 * b21 -> C01
    U[2][8] = 1.0
    V[5][8] = 1.0
    W[2][8] = 1.0

    # a12 * b20 -> C10
    U[5][9] = 1.0
    V[4][9] = 1.0
    W[1][9] = 1.0

    # a12 * b21 -> C11
    U[5][10] = 1.0
    V[5][10] = 1.0
    W[3][10] = 1.0

    return U, V, W

def _build_55_rank():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    block_rank = 11
    total_rank = num_blocks * block_rank

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    Ub, Vb, Wb = _build_block_232()

    for t in range(num_blocks):
        offset = t * block_rank
        for i in range(n * m):
            for c in range(block_rank):
                U[i][offset + c] = Ub[i][c]

        for j in range(m):
            for k_local in range(2):
                k_global = t * 2 + k_local
                v_local_idx = j * 2 + k_local
                v_global_idx = j * p + k_global
                for c in range(block_rank):
                    V[v_global_idx][offset + c] = Vb[v_local_idx][c]

        for k_local in range(2):
            k_global = t * 2 + k_local
            w_global_idx0 = k_global * n + 0
            w_global_idx1 = k_global * n + 1
            for c in range(block_rank):
                W[w_global_idx0][offset + c] = Wb[k_local * 2 + 0][c]
                W[w_global_idx1][offset + c] = Wb[k_local * 2 + 1][c]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_55_rank()
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
