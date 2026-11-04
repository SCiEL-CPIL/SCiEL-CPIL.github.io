import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen factors for 2x2
    # U1, V1, W1 shapes (4, 7)
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices for 2x2: 00, 01, 10, 11
    # U1[i*2+j, c]
    # V1[j*2+k, c]
    # W1[k*2+i, c]
    
    # c=0 (M1)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[1, 0] = 1; W1[2, 0] = 1; W1[3, 0] = 1 # C00+=M1, C01+=M1, C10+=M1, C11+=M1? No.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # Let's fill W1 based on the reconstruction formulas.
    # W1[k*2+i, c] corresponds to C[i, k] coefficient.
    # C00 (i=0, k=0) -> idx 0
    # C01 (i=0, k=1) -> idx 1
    # C10 (i=1, k=0) -> idx 2
    # C11 (i=1, k=1) -> idx 3
    
    # M1 (c=0)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1  # C00
    W1[3, 0] = 1  # C11
    
    # M2 (c=1)
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1  # C10
    
    # M3 (c=2)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1  # C01
    W1[3, 2] = -1 # C11
    
    # M4 (c=3)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = -1 # C00
    W1[2, 3] = 1  # C10
    
    # M5 (c=4)
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = 1  # C00
    W1[1, 4] = 1  # C01
    
    # M6 (c=5)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C11
    
    # M7 (c=6)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = -1 # C00
    W1[3, 6] = -1 # C11
    
    # 2. Schoolbook factors for 3x3
    # U2, V2, W2 shapes (9, 27)
    U2 = np.zeros((9, 27))
    V2 = np.zeros((9, 27))
    W2 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U2[i*3+j, col] = 1
                V2[j*3+k, col] = 1
                W2[k*3+i, col] = 1
                col += 1
                
    # 3. Combine
    # Rank = 7 * 27 = 189
    rank = 7 * 27
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for c1 in range(7):
        for c2 in range(27):
            c = c1 * 27 + c2
            # Kronecker product of vectors
            # U index: (i1*2+j1) * 9 + (i2*3+j2) ?
            # The combined index for U is (i_total * 6 + j_total)
            # i_total = i1 * 3 + i2
            # j_total = j1 * 3 + j2
            # So index = (i1*3+i2)*6 + (j1*3+j2)
            
            for i1 in range(2):
                for j1 in range(2):
                    u1_val = U1[i1*2+j1, c1]
                    if u1_val == 0: continue
                    for i2 in range(3):
                        for j2 in range(3):
                            u2_val = U2[i2*3+j2, c2]
                            if u2_val == 0: continue
                            i_tot = i1 * 3 + i2
                            j_tot = j1 * 3 + j2
                            U[i_tot * 6 + j_tot, c] += u1_val * u2_val
                            
            for j1 in range(2):
                for k1 in range(2):
                    v1_val = V1[j1*2+k1, c1]
                    if v1_val == 0: continue
                    for j2 in range(3):
                        for k2 in range(3):
                            v2_val = V2[j2*3+k2, c2]
                            if v2_val == 0: continue
                            j_tot = j1 * 3 + j2
                            k_tot = k1 * 3 + k2
                            V[j_tot * 6 + k_tot, c] += v1_val * v2_val
                            
            for k1 in range(2):
                for i1 in range(2):
                    w1_val = W1[k1*2+i1, c1]
                    if w1_val == 0: continue
                    for k2 in range(3):
                        for i2 in range(3):
                            w2_val = W2[k2*3+i2, c2]
                            if w2_val == 0: continue
                            k_tot = k1 * 3 + k2
                            i_tot = i1 * 3 + i2
                            W[k_tot * 6 + i_tot, c] += w1_val * w2_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
