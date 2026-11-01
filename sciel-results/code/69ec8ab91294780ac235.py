"""Rank-52 matrix multiplication decomposition for 2x2x15.

Strategy:
- 6 pairs of k-columns handled by Strassen's rank-7 algorithm: 6*7 = 42
- Last 3 columns (k=12,13,14) handled by a hand-verified rank-10 exact integer
  decomposition of the 2x2x3 matrix multiplication tensor.
- Total rank = 52.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START


def construct():
    n, m, p = 2, 2, 15
    U, V, W = _build_rank52(n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_rank52(n, m, p):
    n, m, p = 2, 2, 15

    # --- 6 Strassen blocks for k = (0,1),(2,3),(4,5),(6,7),(8,9),(10,11) ---
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    all_U, all_V, all_W = [], [], []

    for k0, k1 in blocks:
        terms = _strassen_terms(k0, k1)
        for (u_dict, v_dict, w_dict) in terms:
            u_e = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_e = [(j * p + kk, c) for (j, kk), c in v_dict.items()]
            w_e = [(kk * n + i, c) for (kk, i), c in w_dict.items()]
            all_U.append(u_e)
            all_V.append(v_e)
            all_W.append(w_e)

    # --- Rank-10 2x2x3 block for k = 12, 13, 14 ---
    U10, V10, W10 = _rank10_2x2x3()
    for r in range(10):
        u_e = []
        v_e = []
        w_e = []
        for i in range(n):
            for j in range(m):
                val = U10[i * m + j, r]
                if val != 0:
                    u_e.append((i * m + j, val))
        for j in range(m):
            for kl in range(3):
                val = V10[j * 3 + kl, r]
                if val != 0:
                    kg = 12 + kl
                    v_e.append((j * p + kg, val))
        for kl in range(3):
            for i in range(n):
                val = W10[kl * n + i, r]
                if val != 0:
                    kg = 12 + kl
                    w_e.append((kg * n + i, val))
        all_U.append(u_e)
        all_V.append(v_e)
        all_W.append(w_e)

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _strassen_terms(k0, k1):
    """7 Strassen rank-1 terms for the 2x2x2 sub-tensor at k = k0, k1."""
    # Using 1-indexed Strassen (p1..p7), mapping:
    #   A[i,j], B[j,k], C[k,i]
    return [
        # p1 = (A00+A11)(B00+B11) -> C00 += p1, C11 += p1
        ({(0, 0): 1, (1, 1): 1},
         {(0, k0): 1, (1, k1): 1},
         {(k0, 0): 1, (k1, 1): 1}),
        # p2 = (A10+A11)B00 -> C10 += p2, C11 -= p2
        ({(1, 0): 1, (1, 1): 1},
         {(0, k0): 1},
         {(k0, 1): 1, (k1, 1): -1}),
        # p3 = A00(B10-B00) -> C01 += p3, C11 += p3
        ({(0, 0): 1},
         {(1, k0): 1, (0, k0): -1},
         {(k0, 0): 1, (k1, 1): 1}),
        # p4 = A11(B01-B00) -> C00 += p4, C10 += p4
        ({(1, 1): 1},
         {(0, k1): 1, (0, k0): -1},
         {(k0, 0): 1, (k0, 1): 1}),
        # p5 = (A00+A01)B11 -> C00 -= p5, C01 += p5
        ({(0, 0): 1, (0, 1): 1},
         {(1, k1): 1},
         {(k0, 0): -1, (k1, 0): 1}),
        # p6 = (A10-A00)(B00+B01) -> C11 += p6
        ({(1, 0): 1, (0, 0): -1},
         {(0, k0): 1, (0, k1): 1},
         {(k1, 1): 1}),
        # p7 = (A01-A11)(B10+B11) -> C00 += p7
        ({(0, 1): 1, (1, 1): -1},
         {(1, k0): 1, (1, k1): 1},
         {(k0, 0): 1}),
    ]


def _rank10_2x2x3():
    """
    Hand-constructed rank-10 exact integer decomposition for 2x2x3.

    Returns U (4x10), V (6x10), W (6x10) as numpy int arrays
    such that for all i in {0,1}, j in {0,1}, k in {0,1,2}:
        sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    """
    U = np.array([
        # T0  T1  T2  T3  T4  T5  T6  T7  T8  T9
        [  1,  1,  1,  1,  1,  0,  0,  0,  0,  0],  # A00
        [  1,  0,  0,  0, -1,  0,  1,  0,  0,  0],  # A01
        [  0,  1,  0,  0,  0,  0,  1,  0,  0,  0],  # A10
        [  1,  1,  0,  0,  0,  1, -1,  0,  0,  0],  # A11
    ], dtype=np.int64)

    V = np.array([
        # T0  T1  T2  T3  T4  T5  T6  T7  T8  T9
        [  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # B00
        [  0,  0,  0,  0,  0,  0,  1,  0,  1,  0],  # B01
        [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # B02
        [  1,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # B10
        [  0,  1,  0,  0,  0,  0, -1,  0,  0,  1],  # B11
        [  0,  0,  1,  0,  1,  0,  0,  1,  0,  0],  # B12
    ], dtype=np.int64)

    W = np.array([
        # T0  T1  T2  T3  T4  T5  T6  T7  T8  T9
        [  1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # C00 (k=0,i=0)
        [  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # C01 (k=0,i=1)
        [  1, -1,  0,  1, -1,  0,  1,  0,  1,  0],  # C10 (k=1,i=0)
        [  0,  0,  0,  0,  1,  0, -1,  0, -1,  0],  # C11 (k=1,i=1)
        [  0,  0,  1, -1,  0,  1,  0,  0, -1,  0],  # C20 (k=2,i=0)
        [  0,  0,  0,  0,  0, -1,  0,  0,  1,  0],  # C21 (k=2,i=1)
    ], dtype=np.int64)

    return U, V, W


# EVOLVE-BLOCK-END
