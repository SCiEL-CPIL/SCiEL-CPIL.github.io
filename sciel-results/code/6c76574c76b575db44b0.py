import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    
    # Strassen 2x2 decomposition factors
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices for 2x2:
    # U row: i*2 + j for A[i,j]
    # V row: j*2 + k for B[j,k]
    # W row: k*2 + i for C[k,i] (trace convention W[k*n+i])
    
    U_S = np.zeros((4, 7), dtype=int)
    V_S = np.zeros((4, 7), dtype=int)
    W_S = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1
    
    # M3: A00(B01-B11) -> C01
    U_S[0, 2] = 1
    V_S[1, 2] = 1; V_S[3, 2] = -1
    W_S[1, 2] = 1
    
    # M4: A11(B10-B00) -> C10
    U_S[3, 3] = 1
    V_S[2, 3] = 1; V_S[0, 3] = -1
    W_S[2, 3] = 1
    
    # M5: (A00+A10)B11 -> -C00 + C11
    U_S[0, 4] = 1; U_S[2, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[3, 4] = 1
    
    # M6: (A01-A00)(B00+B10) -> C11
    U_S[1, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[2, 5] = 1
    W_S[3, 5] = 1
    
    # M7: (A10-A11)(B11+B01) -> C00
    U_S[2, 6] = 1; U_S[3, 6] = -1
    V_S[3, 6] = 1; V_S[1, 6] = 1
    W_S[0, 6] = 1

    # Kronecker product for 4x4
    # Rank = 49
    rank = 49
    U_kr = np.zeros((16, rank), dtype=int)
    V_kr = np.zeros((16, rank), dtype=int)
    W_kr = np.zeros((16, rank), dtype=int)
    
    # Fill Kronecker product matrices
    # Row index R_kr = i1 * 4 + i2
    for i1 in range(4):
        for i2 in range(4):
            R_kr = i1 * 4 + i2
            for c1 in range(7):
                for c2 in range(7):
                    C_kr = c1 * 7 + c2
                    U_kr[R_kr, C_kr] = U_S[i1, c1] * U_S[i2, c2]
                    V_kr[R_kr, C_kr] = V_S[i1, c1] * V_S[i2, c2]
                    W_kr[R_kr, C_kr] = W_S[i1, c1] * W_S[i2, c2]
                    
    # Permute rows to match 4x4 MM tensor indexing
    # A index: R_mm = r * 4 + c
    # r = 2*r1 + r2, c = 2*c1 + c2
    # i1 = 2*r1 + c1, i2 = 2*r2 + c2
    # R_kr = i1 * 4 + i2
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for r1 in range(2):
        for c1 in range(2):
            for r2 in range(2):
                for c2 in range(2):
                    # For U (A)
                    i1 = 2*r1 + c1
                    i2 = 2*r2 + c2
                    R_kr = i1 * 4 + i2
                    r = 2*r1 + r2
                    c = 2*c1 + c2
                    R_mm = r * 4 + c
                    perm_U[R_mm] = R_kr
                    
    for c1 in range(2):
        for d1 in range(2):
            for c2 in range(2):
                for d2 in range(2):
                    # For V (B)
                    # V index corresponds to B[c, d]
                    # j1 = 2*c1 + d1 (row of outer block, col of inner? No)
                    # In 2x2: V row j = j_row * 2 + j_col? No.
                    # V_S row index is for B[j, k].
                    # j is the shared index.
                    # In Kronecker: j1 is shared index for outer block.
                    # j1 = 2*c1 + d1? 
                    # Outer block B[j1, d1]. j1 is row, d1 is col.
                    # Inner block B[j2, d2].
                    # Shared index c = 2*c1 + c2.
                    # So c1 is the "row" of B in outer block? Yes.
                    # And c1 is the "col" of A in outer block? Yes.
                    # So j1 = 2*c1 + d1 is correct for V's outer index.
                    # j2 = 2*c2 + d2 is correct for V's inner index.
                    j1 = 2*c1 + d1
                    j2 = 2*c2 + d2
                    R_kr_V = j1 * 4 + j2
                    c = 2*c1 + c2
                    d = 2*d1 + d2
                    R_mm_V = c * 4 + d
                    perm_V[R_mm_V] = R_kr_V
                    
    for r1 in range(2):
        for d1 in range(2):
            for r2 in range(2):
                for d2 in range(2):
                    # For W (C)
                    # W index corresponds to C[r, d]
                    # k1 = 2*r1 + d1 (row of C outer, col of C outer? No)
                    # W_S row index is for C[k, i].
                    # k is col, i is row.
                    # In Kronecker: k1 is shared? No.
                    # C[r1, d1] is outer. C[r2, d2] is inner.
                    # k1 = 2*d1 + r1? 
                    # W_S index: k*2 + i.
                    # Outer: k1 = d1, i1 = r1.
                    # So W_S outer index = d1 * 2 + r1.
                    # Inner: k2 = d2, i2 = r2.
                    # W_S inner index = d2 * 2 + r2.
                    k1 = 2*d1 + r1
                    k2 = 2*d2 + r2
                    R_kr_W = k1 * 4 + k2
                    r = 2*r1 + r2
                    d = 2*d1 + d2
                    R_mm_W = d * 4 + r  # Trace convention W[k*n+i] -> W[d*4+r]
                    perm_W[R_mm_W] = R_kr_W

    # Apply permutations
    U = U_kr[perm_U, :]
    V = V_kr[perm_V, :]
    W = W_kr[perm_W, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
