"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Try rank 48 via tensor slicing:
    # Split k into {0,1,2,3} and {4,5,6}
    # Sub-tensor 1: 2x4x4 (k=0..3) -> rank 32 (Strassen on 2x2x2 blocks)
    # Sub-tensor 2: 2x4x3 (k=4..6) -> rank 16 (schoolbook on 2x4x3)
    # Total: 32 + 16 = 48
    
    # Let's try a different slicing strategy to potentially get lower rank.
    # Actually, let's try to use Strassen more cleverly.
    
    # Strategy: Use Strassen on 2x2x2 blocks for the 2x4x4 part (k=0..3)
    # and schoolbook for the 2x4x3 part (k=4..6).
    
    # 2x4x4 tensor:
    # Split j into (0,1) and (2,3). Split k into (0,1) and (2,3).
    # 4 blocks of 2x2x2. Each uses Strassen (rank 7). Total: 4 * 7 = 28.
    
    # 2x4x3 tensor (k=4,5,6):
    # Schoolbook: 2 * 4 * 3 = 24.
    
    # Total: 28 + 24 = 52. Not better than 50.
    
    # Let's stick with the rank 50 approach but verify correctness.
    # Or try to find rank 48 or 49.
    
    # Known lower bounds:
    # R(2x4x7) >= max(2*4, 4*7, 2*7, ceil(2*4*7 / max(2,4,7))) 
    # = max(8, 28, 14, ceil(56/7)) = max(8, 28, 14, 8) = 28.
    # Strassen gives 50.
    
    # Let's try a hybrid:
    # Split k into (0,1,2,3,4) and (5,6)
    # 2x4x5: Use Strassen on 2x2x2 for k=0,1 and k=2,3 (2 blocks * 7 = 14)
    #         and schoolbook for k=4 (2*4*1 = 8). Total: 22.
    # 2x4x2: Use Strassen on 2x2x2 for k=5,6 (1 block * 7 = 7).
    # Total: 22 + 7 = 29. Wait, that's not right.
    # 2x4x5: k=0,1,2,3,4. 
    #   j=(0,1), k=(0,1): Strassen (7)
    #   j=(2,3), k=(0,1): Strassen (7)
    #   j=(0,1), k=(2,3): Strassen (7)
    #   j=(2,3), k=(2,3): Strassen (7)
    #   j=(0,1), k=4: schoolbook (4)
    #   j=(2,3), k=4: schoolbook (4)
    #   Total: 7+7+7+7+4+4 = 36.
    # 2x4x2: k=5,6.
    #   j=(0,1), k=(5,6): Strassen (7)
    #   j=(2,3), k=(5,6): Strassen (7)
    #   Total: 14.
    # Grand total: 36 + 14 = 50. Same as before.
    
    # Let's just return the rank 50 solution.
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for k_start in [0, 2, 4]:
        for j_start in [0, 2]:
            add_strassen_block(0, j_start, k_start)
    
    for j_start in [0, 2]:
        add_schoolbook_221_block(0, j_start, 6)

    if col != rank:
        return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
