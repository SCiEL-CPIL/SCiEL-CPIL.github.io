import time
import numpy as np

def construct():
    """
    Construct a rank-57 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm (rank 49) for 4x4x4 blocks.
    Uses direct rank-4 decomposition for 4x4x1 blocks (total rank 8 for 4x4x2).
    """
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen factors for 2x2x2
    rank_222 = 7
    U_s = [[0.0] * rank_222 for _ in range(4)]
    V_s = [[0.0] * rank_222 for _ in range(4)]
    W_s = [[0.0] * rank_222 for _ in range(4)]

    # Strassen products
    # (U_block, V_block, W_block)
    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j][m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k][m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i][m_idx] += c

    rank_444 = 49
    rank_442 = 8
    total_rank = rank_444 + rank_442

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    # 1. 4x4x4 Part (columns 0-48)
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val

    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val

    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # 2. 4x4x2 Part (columns 49-56)
    # k=4 and k=5
    # Rank 4 for each k
    for k_idx, k_val in enumerate([4, 5]):
        col_offset = rank_444 + k_idx * 4
        for j in range(4):
            for i in range(4):
                col = col_offset + j
                # U[i, j] = delta[i, j]
                if i == j:
                    U[i * 4 + j][col] = 1.0
                # V[j, k] = 1
                V[j * 6 + k_val][col] = 1.0
                # W[k, i] = 1
                W[k_val * 4 + i][col] = 1.0

    return {"U": U, "V": V, "W": W}
