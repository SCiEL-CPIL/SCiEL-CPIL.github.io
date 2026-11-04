"""
Matrix multiplication 4x4x6 tensor decomposition.
Mixed approach: 4x4x4 Strassen (rank 49) for first 4 columns of B,
and four independent 2x2x2 Strassen blocks (rank 7 each = 28 total)
for the last 2 columns of B. Total rank = 77.
"""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np


def _strassen_2x2():
    """Return 2x2x2 Strassen factors. U:(4,7), V:(4,7), W:(4,7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]."""
    r = 7
    U = [[0.0] * r for _ in range(4)]
    V = [[0.0] * r for _ in range(4)]
    W = [[0.0] * r for _ in range(4)]
    prods = [
        ([(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)]),
        ([(1, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0)], [(1, 0, 1.0), (1, 1, -1.0)]),
        ([(0, 0, 1.0)], [(0, 1, 1.0), (1, 1, -1.0)], [(0, 1, 1.0), (1, 1, 1.0)]),
        ([(1, 1, 1.0)], [(1, 0, 1.0), (0, 0, -1.0)], [(0, 0, 1.0), (1, 0, 1.0)]),
        ([(0, 0, 1.0), (0, 1, 1.0)], [(1, 1, 1.0)], [(0, 0, -1.0), (0, 1, 1.0)]),
        ([(1, 0, 1.0), (0, 0, -1.0)], [(0, 0, 1.0), (0, 1, 1.0)], [(1, 1, 1.0)]),
        ([(0, 1, 1.0), (1, 1, -1.0)], [(1, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0)]),
    ]
    for c, (ub, vb, wb) in enumerate(prods):
        for (i, j, cv) in ub:
            U[i * 2 + j][c] += cv
        for (j, k, cv) in vb:
            V[j * 2 + k][c] += cv
        for (i, k, cv) in wb:
            W[k * 2 + i][c] += cv
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 140.0

    # Fallback: schoolbook rank 96
    def _fallback():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline - 5.0:
        return _fallback()

    Us, Vs, Ws = _strassen_2x2()

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]  # 24 x 77

    # Part 1: 4x4x4 Strassen via Kronecker product of 2x2 factors
    # U4[i*4+j, 7*r1+r2] = Us[(i//2)*2+(j//2), r1] * Us[(i%2)*2+(j%2), r2]
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for r1 in range(7):
                for r2 in range(7):
                    col = 7 * r1 + r2
                    val = Us[I * 2 + J][r1] * Us[i0 * 2 + j0][r2]
                    if val != 0:
                        U[i * 4 + j][col] += val

    # V4[j*4+k, 7*r1+r2] = Vs[(j//2)*2+(k//2), r1] * Vs[(j%2)*2+(k%2), r2]
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for r1 in range(7):
                for r2 in range(7):
                    col = 7 * r1 + r2
                    val = Vs[J * 2 + K][r1] * Vs[j0 * 2 + k0][r2]
                    if val != 0:
                        V[j * 6 + k][col] += val

    # W4[k*4+i, 7*r1+r2] = Ws[(k//2)*2+(i//2), r1] * Ws[(k%2)*2+(i%2), r2]
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for r1 in range(7):
                for r2 in range(7):
                    col = 7 * r1 + r2
                    val = Ws[K * 2 + I][r1] * Ws[k0 * 2 + i0][r2]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # Part 2: Four 2x2x2 Strassen blocks for columns 4-5 of B
    # Block (I, J): A[2I:2I+2, 2J:2J+2] * B[2J:2J+2, 4:6] -> C[2I:2I+2, 4:6]
    # where J=0 uses B rows 0-1 (top), J=1 uses B rows 2-3 (bot)
    blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for b_idx, (I, J) in enumerate(blocks):
        col_off = rank_444 + b_idx * 7
        j_base = 2 * J  # 0 for top, 2 for bot
        i_base = 2 * I
        k_base = 4

        # U part: A block
        for i0 in range(2):
            for j0 in range(2):
                i_glob = i_base + i0
                j_glob = j_base + j0
                for s in range(7):
                    v = Us[i0 * 2 + j0][s]
                    if v != 0:
                        U[i_glob * 4 + j_glob][col_off + s] += v

        # V part: B block (rows j_base..j_base+1, cols 4..5)
        for j0 in range(2):
            for k0 in range(2):
                j_glob = j_base + j0
                k_glob = k_base + k0
                for s in range(7):
                    v = Vs[j0 * 2 + k0][s]
                    if v != 0:
                        V[j_glob * 6 + k_glob][col_off + s] += v

        # W part: C block (rows i_base..i_base+1, cols 4..5)
        for k0 in range(2):
            for i0 in range(2):
                i_glob = i_base + i0
                k_glob = k_base + k0
                for s in range(7):
                    v = Ws[k0 * 2 + i0][s]
                    if v != 0:
                        W[k_glob * 4 + i_glob][col_off + s] += v

    if time.time() > deadline - 8.0:
        return {"U": U, "V": V, "W": W}

    # Verify
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, total_rank)
    Vr = Va.reshape(m, p, total_rank)
    Wr = Wa.reshape(p, n, total_rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = float(np.max(np.abs(T - 1.0)))

    if err > 1e-6:
        return _fallback()

    return {"U": U, "V": V, "W": W}
