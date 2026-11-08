import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    # Rank 50 decomposition for 2x4 * 4x7
    # Structure:
    # A is 2x4, B is 4x7.
    # Split A into two 2x2 blocks: A1, A2 (columns 0-1, 2-3)
    # Split B into two 2x7 blocks: B1, B2 (rows 0-1, 2-3)
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x7.
    # Split B_i into three 2x2 and one 2x1.
    # 3 * Strassen(2x2x2) [rank 7] + 1 * Schoolbook(2x2x1) [rank 4] = 25.
    # Total rank = 2 * 25 = 50.

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices: W[k*n + i]
        # Output C[i, k]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # Strassen 7 multiplications
        # M1 = (a+b)(e+h)
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c01_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (a-b)(e-h)
        U[a_idx][col] += 1
        U[b_idx][col] -= 1
        V[e_idx][col] += 1
        V[h_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c01_idx][col] -= 1
        W[c10_idx][col] -= 1
        W[c11_idx][col] += 1
        col += 1
        
        # M3 = (c+d)(e+h)
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] -= 1
        W[c10_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = (c-d)(e-h)
        U[c_idx][col] += 1
        U[d_idx][col] -= 1
        V[e_idx][col] += 1
        V[h_idx][col] -= 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M5 = a(i-j)
        U[a_idx][col] += 1
        V[g_idx][col] += 1
        V[f_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M6 = (a+b)(j-h)
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        V[h_idx][col] -= 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M7 = (c+d)(i-j)
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[f_idx][col] -= 1
        W[c10_idx][col] -= 1
        W[c11_idx][col] += 1
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
        
        # a*e
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # b*g
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # c*e
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # d*g
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Apply blocks
    # j0 in 0, 2 (for A columns)
    # k0 in 0, 2, 4 (Strassen) and 6 (Schoolbook)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
        add_schoolbook_221_block(0, j0, 6)

    return {"U": U, "V": V, "W": W}
