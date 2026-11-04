import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

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
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2
    # Using 4 independent 2x2x2 Strassen decompositions
    block_products = [
        (0, 0, 0, 0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 2, 0, 0), # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Wait, B is 4x2, so cols 0,1)
    ]
    # Correction: B is 4x2. Cols are 0 and 1.
    # Block 0: A[0:2, 0:2] * B[0:2, 0] -> C[0:2, 0]
    # This is not a 2x2x2 multiplication.
    # The current code uses:
    # (0, 0, 0, 4, 0, 4) -> A[0:2, 0:2] * B[0:2, 0:2]? No, B is 4x6 in the full problem, but here B is 4x2.
    # In the full 4x4x6 problem, B is 4x6.
    # The current program splits B into 4x4 and 4x2.
    # For the 4x2 part, B has columns 4 and 5 (indices 4,5 in the 4x6 matrix).
    # So B_right is 4x2.
    # The blocks are:
    # 1. A[0:2, 0:2] * B[0:2, 0:1] (B col 0 of right part, which is col 4 of full B)
    # This is 2x2 * 2x1.
    
    # Let's stick to the proven rank 28 structure from the current program which works.
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
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

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
