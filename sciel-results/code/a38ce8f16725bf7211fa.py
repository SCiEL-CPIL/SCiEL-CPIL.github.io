import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00(-), C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # Local indices for 2x2 block:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: A00, A01, A10, A11
        # V: B00, B01, B10, B11
        # W: C00, C01, C10, C11 (indexed by k*n+i -> 0, 1, 2, 3 for n=2)
        # W index: k*2+i. C00->0, C01->1, C10->2, C11->3.
        # Wait, W index in code is k*n+i.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2? No.
        # C is n x p. Rows i, Cols k.
        # W index is k*n + i.
        # So C[i,k] is at index k*n + i.
        # C00: i=0, k=0 -> 0
        # C01: i=0, k=1 -> 2
        # C10: i=1, k=0 -> 1
        # C11: i=1, k=1 -> 3
        
        # M1: C00(+1), C11(+1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: C10(+1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: C01(+1), C11(+1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4: C00(+1), C10(+1)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5: C00(-1), C01(+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: C11(+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: C00(+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Blocks:
    # 0: A_L B_T0. A cols 0,1. B rows 0,1, cols 0,1. C cols 0,1.
    # 1: A_L B_T1. A cols 0,1. B rows 0,1, cols 2,3. C cols 2,3.
    # 2: A_L B_T2. A cols 0,1. B rows 0,1, cols 4,5. C cols 4,5.
    # 3: A_R B_B0. A cols 2,3. B rows 2,3, cols 0,1. C cols 0,1.
    # 4: A_R B_B1. A cols 2,3. B rows 2,3, cols 2,3. C cols 2,3.
    # 5: A_R B_B2. A cols 2,3. B rows 2,3, cols 4,5. C cols 4,5.
    
    blocks = [
        # (i_offset, j_offset, k_offset)
        # i is row of A (0,1). j is col of A / row of B. k is col of B.
        # A_L: j_offset 0. A_R: j_offset 2.
        # B_T: j_offset 0 (rows 0,1). B_B: j_offset 2 (rows 2,3).
        # Cols of B: 0, 2, 4.
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        for u_c, v_c, w_c in strassen_terms:
            # u_c: [A00, A01, A10, A11]
            # Local A indices:
            # A00: i=0, j=0 -> global i=i_off, j=j_off
            # A01: i=0, j=1 -> global i=i_off, j=j_off+1
            # A10: i=1, j=0 -> global i=i_off+1, j=j_off
            # A11: i=1, j=1 -> global i=i_off+1, j=j_off+1
            
            U[i_off * m + j_off][col] += u_c[0]
            U[i_off * m + j_off + 1][col] += u_c[1]
            U[(i_off + 1) * m + j_off][col] += u_c[2]
            U[(i_off + 1) * m + j_off + 1][col] += u_c[3]
            
            # v_c: [B00, B01, B10, B11]
            # Local B indices:
            # B00: j=0, k=0 -> global j=j_off, k=k_off
            # B01: j=0, k=1 -> global j=j_off, k=k_off+1
            # B10: j=1, k=0 -> global j=j_off+1, k=k_off
            # B11: j=1, k=1 -> global j=j_off+1, k=k_off+1
            
            V[j_off * p + k_off][col] += v_c[0]
            V[j_off * p + k_off + 1][col] += v_c[1]
            V[(j_off + 1) * p + k_off][col] += v_c[2]
            V[(j_off + 1) * p + k_off + 1][col] += v_c[3]
            
            # w_c: [C00, C01, C10, C11]
            # Local C indices (i, k):
            # C00: i=0, k=0 -> global i=i_off, k=k_off
            # C01: i=0, k=1 -> global i=i_off, k=k_off+1
            # C10: i=1, k=0 -> global i=i_off+1, k=k_off
            # C11: i=1, k=1 -> global i=i_off+1, k=k_off+1
            
            # W index: k_global * n + i_global
            W[k_off * n + i_off][col] += w_c[0]
            W[(k_off + 1) * n + i_off][col] += w_c[1]
            W[k_off * n + (i_off + 1)][col] += w_c[2]
            W[(k_off + 1) * n + (i_off + 1)][col] += w_c[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
