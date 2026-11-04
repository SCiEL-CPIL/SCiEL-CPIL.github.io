"""
Strategy: Decompose 2x3x8 as 4 independent 2x3x2 blocks.
Each 2x3x2 = 2x2x2 (Strassen, rank 7) + 2x1x2 (schoolbook, rank 2) = rank 9.
Total rank: 4 * 9 = 36.
"""
import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 2, 3, 8
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 16 x 36

    # Strassen 2x2x2 matrices
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    col = 0
    for block in range(4):
        k0 = block * 2  # starting k index for this block
        
        # Strassen part: 2x2x2 for a in {0,1}, i in {0,1}, k in {k0, k0+1}
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1

        # Schoolbook part: 2x1x2 for a=2, i in {0,1}, k in {k0, k0+1}
        for li in range(2):
            for lk in range(2):
                U[li * m + 2][col] = 1
                V[2 * p + (k0 + lk)][col] = 1
                W[(k0 + lk) * n + li][col] = 1
                col += 1

    candidate = {"U": U, "V": V, "W": W}

    # Validate
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                for c in range(rank):
                    acc += U[i * m + a][c] * V[a * p + k][c] * W[k * n + i][c]
                if acc != 1:
                    # Fallback to schoolbook
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    col_sb = 0
                    for ii in range(n):
                        for aa in range(m):
                            for kk in range(p):
                                U_sb[ii * m + aa][col_sb] = 1
                                V_sb[aa * p + kk][col_sb] = 1
                                W_sb[kk * n + ii][col_sb] = 1
                                col_sb += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}

    return candidate
