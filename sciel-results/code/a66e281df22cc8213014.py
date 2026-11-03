import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_222():
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U: 2i+j, V: 2j+k, W: 2k+i
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1 # A11 (0), A22 (3)
    V[0, 0] = 1; V[3, 0] = 1 # B11 (0), B22 (3)
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C11, C12, C21, C22 all get +1? 
    # Wait, W entries are coefficients for C.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define W based on the formulas.
    # W[row, col] is the coefficient of M_col in C_row.
    # Row 0: C11 (k=0, i=0) -> W[0, :]
    # Row 1: C12 (k=1, i=0) -> W[1, :]
    # Row 2: C21 (k=0, i=1) -> W[2, :]
    # Row 3: C22 (k=1, i=1) -> W[3, :]
    
    W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1 # M1 in C11, C22
    W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = -1 # M2 in C21, C22
    W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 0; W[3, 2] = 1 # M3 in C12, C22
    W[0, 3] = 1; W[1, 3] = 0; W[2, 3] = 1; W[3, 3] = 0 # M4 in C11, C21
    W[0, 4] = -1; W[1, 4] = 1; W[2, 4] = 0; W[3, 4] = 0 # M5 in C11, C12
    W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = 1 # M6 in C22
    W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0 # M7 in C11
    
    # Now U and V for each M
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # 1. Get 2x2x2 Strassen
    U1, V1, W1 = _strassen_222()
    
    # 2. Construct 4x4x4 part (Rank 49)
    # We need to map indices.
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            # U: index 4i+j
            # i = 2i1 + i2, j = 2j1 + j2
            # U1 index r1 = 2i1 + j1
            # U2 index r2 = 2i2 + j2
            # We need to iterate over all non-zero entries in U1[:, t1] and U2[:, t2]
            nz1 = np.nonzero(U1[:, t1])[0]
            nz2 = np.nonzero(U2 := U1[:, t2])[0] # Reusing U1 for U2 since it's the same structure
            
            for r1 in nz1:
                # r1 = 2i1 + j1
                i1 = r1 // 2
                j1 = r1 % 2
                val1 = U1[r1, t1]
                for r2 in nz2:
                    # r2 = 2i2 + j2
                    i2 = r2 // 2
                    j2 = r2 % 2
                    val2 = U1[r2, t2]
                    
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    r4 = 4*i + j
                    U4[r4, t] += val1 * val2
                    
    # V: index 4j+k
    # j = 2j1 + j2, k = 2k1 + k2
    # V1 index r1 = 2j1 + k1
    # V2 index r2 = 2j2 + k2
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            nz1 = np.nonzero(V1[:, t1])[0]
            nz2 = np.nonzero(V1[:, t2])[0]
            
            for r1 in nz1:
                j1 = r1 // 2
                k1 = r1 % 2
                val1 = V1[r1, t1]
                for r2 in nz2:
                    j2 = r2 // 2
                    k2 = r2 % 2
                    val2 = V1[r2, t2]
                    
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    r4 = 4*j + k
                    V4[r4, t] += val1 * val2
                    
    # W: index 4k+i
    # k = 2k1 + k2, i = 2i1 + i2
    # W1 index r1 = 2k1 + i1
    # W2 index r2 = 2k2 + i2
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            nz1 = np.nonzero(W1[:, t1])[0]
            nz2 = np.nonzero(W1[:, t2])[0]
            
            for r1 in nz1:
                k1 = r1 // 2
                i1 = r1 % 2
                val1 = W1[r1, t1]
                for r2 in nz2:
                    k2 = r2 // 2
                    i2 = r2 % 2
                    val2 = W1[r2, t2]
                    
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    r4 = 4*k + i
                    W4[r4, t] += val1 * val2
                    
    # 3. Construct 4x4x1 part for column 4 (Rank 16)
    U5 = np.zeros((16, 16), dtype=float)
    V5 = np.zeros((4, 16), dtype=float)
    W5 = np.zeros((16, 16), dtype=float) # Only rows 16-19 used, but we will append to W4 which is 16x49? No, W4 is 16x49 for 4x4x4.
    # Wait, W for 4x4x4 has shape (4*4, 49) = (16, 49).
    # W for 4x4x1 has shape (4*4, 16) = (16, 16).
    # But the full W has shape (5*4, Rank) = (20, Rank).
    # The 4x4x4 part uses rows 0-15 (k=0,1,2,3).
    # The 4x4x1 part uses rows 16-19 (k=4).
    
    for i in range(4):
        for j in range(4):
            t = i * 4 + j
            U5[4*i+j, t] = 1
            V5[j, t] = 1
            # W row index: k=4, i=i. Index = 4*4 + i = 16 + i.
            # But W5 is local. We will place it in the final W.
            W5[16+i, t] = 1
            
    # 4. Combine
    U = np.hstack([U4, U5])
    V = np.hstack([V4, V5])
    W = np.zeros((20, 65), dtype=float)
    W[:16, :49] = W4
    W[:, 49:] = W5
    
    # Convert to lists of floats/ints as required? 
    # "integer/half-integer entries"
    # The values are integers.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
