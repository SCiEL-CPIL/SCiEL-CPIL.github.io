import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _get_strassen_2x2_factors():
    """Returns U, V, W for 2x2x2 Strassen. U,V,W shape (4,7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

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
            U[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W[k * 2 + i, m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0))

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _get_strassen_2x2_factors()

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    # Part 1: 4x4x4 recursive Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I, i0 = i // 2, i % 2
                for j in range(4):
                    J, j0 = j // 2, j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J, j0 = j // 2, j % 2
                for k in range(4):
                    K, k0 = k // 2, k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K, k0 = k // 2, k % 2
                for i in range(4):
                    I, i0 = i // 2, i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4x2 block Strassen
    # 4 blocks of 2x2x2 Strassen
    # A blocks: (0,0), (0,2), (2,0), (2,2)
    # B blocks: (0,4), (0,4), (2,4), (2,4) -> Wait, B is 4x6.
    # B rows 0,1 use B[0:2, 4:6]. B rows 2,3 use B[2:4, 4:6].
    # C blocks: (0,4), (0,4), (2,4), (2,4) -> C rows 0,1 use C[0:2, 4:6]. C rows 2,3 use C[2:4, 4:6].
    
    blocks = [
        (0, 0, 0, 4, 0, 4), # A[0:2,0:2], B[0:2,4:6], C[0:2,4:6]
        (0, 2, 0, 4, 0, 4), # A[0:2,2:4], B[0:2,4:6], C[0:2,4:6]
        (2, 0, 2, 4, 2, 4), # A[2:4,0:2], B[2:4,4:6], C[2:4,4:6]
        (2, 2, 2, 4, 2, 4), # A[2:4,2:4], B[2:4,4:6], C[2:4,4:6]
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    err = _verify(U, V, W, n, m, p, total_rank)

    if err > 1e-6 or time.time() > deadline:
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=float)
        V_f = np.zeros((m * p, rank), dtype=float)
        W_f = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
