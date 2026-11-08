import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Uses Strassen 2x2x2 for the first 6 k-blocks (3 blocks of 2 in k-dimension, 2 blocks of 2 in j-dimension)
    and schoolbook for the last k=6 terms.
    Total rank = 6 * 7 + 8 = 50.
    """
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    rank = 50
    U = [[0] * rank for _ in range(nA)]
    V = [[0] * rank for _ in range(nB)]
    W = [[0] * rank for _ in range(nW)]
    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A indices for the 2x2 block at rows i0..i0+1, cols j0..j0+1
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices for the 2x2 block at rows j0..j0+1, cols k0..k0+1
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices for the 2x2 block in C at rows k0..k0+1, cols i0..i0+1
        # Convention: W[k*n + i, r]
        c00 = k0 * n + i0
        c01 = k0 * n + (i0 + 1)
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> contributes to c00 and c11
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

        # M2 = (c+d)e -> contributes to c10 and c11 (c11 gets -M2)
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1

        # M3 = a(f-h) -> contributes to c01 and c11
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1

        # M4 = d(g-e) -> contributes to c00 and c10
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1

        # M5 = (a+b)h -> contributes to c00 (negative) and c01
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> contributes to c11
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> contributes to c00
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] += 1
        col += 1

    # Apply Strassen to the 6 blocks of 2x2:
    # j_blocks: 2 blocks (cols 0-1, 2-3)
    # k_blocks: 3 blocks (cols 0-1, 2-3, 4-5)
    # i is always 0-1 (full 2 rows)
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)

    # Handle the remaining k=6 column with schoolbook multiplication
    # C[i, 6] = sum_{j=0}^{3} A[i, j] * B[j, 6]
    # This is a 2x4 * 4x1 multiplication, rank 8
    for j in range(m):
        for i in range(n):
            u_idx = i * m + j
            v_idx = j * p + 6
            w_idx = 6 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
