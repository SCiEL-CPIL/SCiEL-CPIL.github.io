import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    
    # Helper to add Strassen 2x2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # Local indices for A: (i, j), i,j in 0,1
    # Local indices for B: (j, k), j,k in 0,1
    # Local indices for C: (i, k), i,k in 0,1
    # Global offsets:
    # A offset: (i_off, j_off)
    # B offset: (j_off, k_off)
    # C offset: (i_off, k_off)
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal r
        # M1 = (A11 + A22)(B11 + B22)
        # A11: i=0, j=0. A22: i=1, j=1.
        # B11: j=0, k=0. B22: j=1, k=1.
        # U entries for A11, A22
        U[i_off*2 + 0, r] = 1 # A11
        U[i_off*2 + 1, r] = 1 # A22
        V[(j_off+0)*5 + k_off, r] = 1 # B11 (j=0, k=0)
        V[(j_off+1)*5 + (k_off+1), r] = 1 # B22 (j=1, k=1)
        W[(k_off+0)*2 + i_off, r] = 1 # C11 part (i=0, k=0)
        W[(k_off+1)*2 + (i_off+1), r] = 1 # C22 part (i=1, k=1)
        # Wait, W index is k*n + i.
        # C11 is i=0, k=0. Index 0*2+0 = 0.
        # C22 is i=1, k=1. Index 1*2+1 = 3.
        # We need to add to C11 and C22.
        # Strassen C11 = M1 + M4 - M5 + M7
        # Strassen C22 = M1 - M2 + M3 + M6
        # So M1 contributes to C11 and C22.
        # My W assignment above is correct for M1.
        r += 1
        
        # M2 = (A21 + A22)B11
        U[i_off*2 + 1, r] = 1 # A21 (i=1, j=0)
        U[i_off*2 + 1, r] = 1 # A22 (i=1, j=1) -> Wait, U[i*m+j].
        # A21: i=1, j=0. Index 1*2+0 = 2.
        # A22: i=1, j=1. Index 1*2+1 = 3.
        U[2, r] = 1
        U[3, r] = 1
        V[(j_off+0)*5 + k_off, r] = 1 # B11
        # C21 = M2 + M4. C22 = M1 - M2 ...
        # M2 contributes to C21 (i=1, k=0) and C22 (i=1, k=1) with -1.
        W[(k_off+0)*2 + (i_off+1), r] = 1 # C21
        W[(k_off+1)*2 + (i_off+1), r] = -1 # C22
        r += 1
        
        # M3 = A11(B12 - B22)
        U[i_off*2 + 0, r] = 1 # A11
        V[(j_off+0)*5 + (k_off+1), r] = 1 # B12 (j=0, k=1)
        V[(j_off+1)*5 + (k_off+1), r] = -1 # B22 (j=1, k=1)
        # C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
        # M3 contributes to C12 (i=0, k=1) and C22 (i=1, k=1).
        W[(k_off+1)*2 + i_off, r] = 1 # C12
        W[(k_off+1)*2 + (i_off+1), r] = 1 # C22
        r += 1
        
        # M4 = A22(B21 - B11)
        U[i_off*2 + 1, r] = 1 # A22
        V[(j_off+1)*5 + k_off, r] = 1 # B21 (j=1, k=0)
        V[(j_off+0)*5 + k_off, r] = -1 # B11 (j=0, k=0)
        # C11 = M1 + M4 ... C21 = M2 + M4.
        # M4 contributes to C11 (i=0, k=0) and C21 (i=1, k=0).
        W[(k_off+0)*2 + i_off, r] = 1 # C11
        W[(k_off+0)*2 + (i_off+1), r] = 1 # C21
        r += 1
        
        # M5 = (A11 + A12)B22
        U[i_off*2 + 0, r] = 1 # A11
        U[i_off*2 + 1, r] = 1 # A12 (i=0, j=1)
        V[(j_off+1)*5 + (k_off+1), r] = 1 # B22
        # C11 = M1 + M4 - M5 ... C12 = M3 + M5.
        # M5 contributes to C11 (i=0, k=0) with -1 and C12 (i=0, k=1) with 1.
        W[(k_off+0)*2 + i_off, r] = -1 # C11
        W[(k_off+1)*2 + i_off, r] = 1 # C12
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[i_off*2 + 1, r] = 1 # A21
        U[i_off*2 + 0, r] = -1 # A11
        V[(j_off+0)*5 + k_off, r] = 1 # B11
        V[(j_off+0)*5 + (k_off+1), r] = 1 # B12
        # C22 = M1 - M2 + M3 + M6.
        # M6 contributes to C22 (i=1, k=1) with 1.
        W[(k_off+1)*2 + (i_off+1), r] = 1 # C22
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[i_off*2 + 1, r] = 1 # A12
        U[i_off*2 + 1, r] = -1 # A22
        V[(j_off+1)*5 + k_off, r] = 1 # B21
        V[(j_off+1)*5 + (k_off+1), r] = 1 # B22
        # C11 = M1 + M4 - M5 + M7.
        # M7 contributes to C11 (i=0, k=0) with 1.
        W[(k_off+0)*2 + i_off, r] = 1 # C11
        r += 1

    # Block 1.1: A1 B11
    # A1: i in 0,1, j in 0,1. Offset i=0, j=0.
    # B11: j in 0,1, k in 0,1. Offset j=0, k=0.
    # C11: i in 0,1, k in 0,1. Offset i=0, k=0.
    add_strassen(0, 0, 0)
    
    # Block 1.2: A1 B12
    # A1: i in 0,1, j in 0,1. Offset i=0, j=0.
    # B12: j in 0,1, k in 2,3. Offset j=0, k=2.
    # C12: i in 0,1, k in 2,3. Offset i=0, k=2.
    add_strassen(0, 0, 2)
    
    # Block 1.3: A1 B13
    # A1: i in 0,1, j in 0,1.
    # B13: j in 0,1, k in 4.
    # C13: i in 0,1, k in 4.
    # Rank 4.
    # C_0,4 = A_0,0 B_0,4 + A_0,1 B_1,4
    # C_1,4 = A_1,0 B_0,4 + A_1,1 B_1,4
    
    # Term 1: A_0,0 B_0,4
    U[0, r] = 1
    V[0*5 + 4, r] = 1
    W[4*2 + 0, r] = 1
    r += 1
    
    # Term 2: A_0,1 B_1,4
    U[1, r] = 1
    V[1*5 + 4, r] = 1
    W[4*2 + 0, r] = 1
    r += 1
    
    # Term 3: A_1,0 B_0,4
    U[2, r] = 1
    V[0*5 + 4, r] = 1
    W[4*2 + 1, r] = 1
    r += 1
    
    # Term 4: A_1,1 B_1,4
    U[3, r] = 1
    V[1*5 + 4, r] = 1
    W[4*2 + 1, r] = 1
    r += 1
    
    # Block 2: A2 B2
    # A2: i in 0,1, j in 2.
    # B2: j in 2, k in 0,1,2,3,4.
    # C2: i in 0,1, k in 0,1,2,3,4.
    # Rank 10.
    for i in range(2):
        for k in range(5):
            U[i*3 + 2, r] = 1
            V[2*5 + k, r] = 1
            W[k*2 + i, r] = 1
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
