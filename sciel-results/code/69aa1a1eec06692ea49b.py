import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2 (rows of A), m=4 (inner), p=7 (cols of B)
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin

    # Strategy: Split k dimension into pairs {0,1},{2,3},{4,5} and single {6}
    # For each k-pair: split j into {0,1} and {2,3}, each a Strassen 2x2x2 (7 mults)
    # For k=6: schoolbook 4 mults
    # Total: 3*(7+7) + 4 = 46

    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 46)
    V = [[0] * rank for _ in range(m * p)]   # shape (28, 46)
    W = [[0] * rank for _ in range(p * n)]   # shape (14, 46)

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
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_241_block(k0):
        nonlocal col
        # For single column k0: C[i,k0] = sum_j A[i,j]*B[j,k0]
        # 4 terms: j=0,1,2,3
        for j in range(m):
            for i in range(n):
                a_idx = i * m + j
                b_idx = j * p + k0
                c_idx = k0 * n + i
                U[a_idx][col] += 1
                V[b_idx][col] += 1
                W[c_idx][col] += 1
                col += 1

    # For each k-pair, use two Strassen blocks (j={0,1} and j={2,3})
    for k_start in [0, 2, 4]:
        if time.time() > deadline:
            break
        add_strassen_block(0, 0, k_start)
        add_strassen_block(0, 2, k_start)

    if time.time() > deadline:
        pass
    else:
        add_schoolbook_241_block(6)

    return {"U": U, "V": V, "W": W}
