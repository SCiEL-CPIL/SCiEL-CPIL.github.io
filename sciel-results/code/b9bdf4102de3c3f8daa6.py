import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    # We use the block decomposition:
    # m=4 split into two blocks of 2.
    # p=7 split into three blocks of 2 and one block of 1.
    # Each 2x2x2 block uses Strassen (rank 7).
    # Each 2x2x1 block uses Schoolbook (rank 4).
    # Total rank = 2 * (3 * 7 + 4) = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # i0: start row of A (0 or 1) - but here A is 2x2, so i0 is always 0 relative to the 2x2 block?
        # Wait, the A block is 2x2. The rows are 0 and 1.
        # The j0 is the start column of A (0 or 2).
        # The k0 is the start column of B (0, 2, 4).
        
        # Local indices for A (2x2)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # Local indices for B (2x2)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # Local indices for C (2x2)
        # C is p x n. Index is k * n + i.
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> c00 += M1, c11 += M1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> c10 += M2, c11 -= M2
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> c01 += M3, c11 += M3
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> c00 += M4, c10 += M4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> c00 -= M5, c01 += M5
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> c11 += M6
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> c00 += M7
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c0_idx = k0 * n + i0
        c1_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> c0 += P1
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c0_idx][col] += 1
        col += 1
        
        # P2 = b*g -> c0 += P2
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c0_idx][col] += 1
        col += 1
        
        # P3 = c*e -> c1 += P3
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c1_idx][col] += 1
        col += 1
        
        # P4 = d*g -> c1 += P4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c1_idx][col] += 1
        col += 1

    # Process the two m-blocks (j=0..1 and j=2..3)
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        # Process the three 2x2x2 k-blocks
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                break
            add_strassen_block(0, j_start, k_start)
        # Process the one 2x2x1 k-block
        if time.time() > deadline:
            break
        add_schoolbook_221_block(0, j_start, 6)
        
    return {"U": U, "V": V, "W": W}
