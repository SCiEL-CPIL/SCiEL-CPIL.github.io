import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 65
    # Part 1: 4x4x4 multiplication using Kronecker Strassen (Rank 49)
    # Part 2: 4x4x1 multiplication using Schoolbook (Rank 16)
    
    # Strassen 2x2x2 decomposition
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    # A11=0,0; A22=1,1; B11=0,0; B22=1,1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # C11, C22 contributions? 
    # Wait, W index is k*2+i.
    # M1 contributes to C11 (k=0,i=0) and C22 (k=1,i=1)?
    # C11 = M1 + ...
    # C22 = M1 - ...
    # So W_s[0*2+0] = 1, W_s[1*2+1] = 1.
    W_s[0, 0] = 1
    W_s[3, 0] = 1
    
    # M2 = (A11+A22)B21
    # A11=0,0; A22=1,1; B21=1,0
    U_s[0, 1] = 1; U_s[3, 1] = 1
    V_s[2, 1] = 1 # j=1, k=0 -> index 1*2+0 = 2
    W_s[2, 1] = 1 # k=0, i=1? No.
    # M2 contributes to C21 (k=0, i=1).
    # W index k*2+i = 0*2+1 = 1.
    W_s[1, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11=0,0; B12=0,1; B22=1,1
    U_s[0, 2] = 1
    V_s[1, 2] = 1 # j=0, k=1 -> 0*2+1 = 1
    V_s[3, 2] = -1 # j=1, k=1 -> 1*2+1 = 3
    # M3 contributes to C12 (k=1, i=0).
    # W index k*2+i = 1*2+0 = 2.
    W_s[2, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22=1,1; B21=1,0; B11=0,0
    U_s[3, 3] = 1
    V_s[2, 3] = 1 # j=1, k=0 -> 2
    V_s[0, 3] = -1 # j=0, k=0 -> 0
    # M4 contributes to C21 (k=0, i=1) and C11 (k=0, i=0)?
    # C21 = M2 + M4. C11 = M1 + M4 - M5 + M7.
    # So W for M4:
    # C21: k=0, i=1 -> idx 1.
    # C11: k=0, i=0 -> idx 0.
    W_s[1, 3] = 1
    W_s[0, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11=0,0; A12=0,1; B22=1,1
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1 # j=1, k=1 -> 3
    # M5 contributes to C12 (k=1, i=0) and C11 (k=0, i=0)?
    # C12 = M3 + M5. C11 = ... - M5 ...
    # So W for M5:
    # C12: k=1, i=0 -> idx 2.
    # C11: k=0, i=0 -> idx 0.
    W_s[2, 4] = 1
    W_s[0, 4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=1,0; A11=0,0; B11=0,0; B12=0,1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    # M6 contributes to C22 (k=1, i=1).
    # C22 = M1 - M2 + M3 + M6.
    # W for M6:
    # C22: k=1, i=1 -> idx 3.
    W_s[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=0,1; A22=1,1; B21=1,0; B22=1,1
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    # M7 contributes to C11 (k=0, i=0).
    # C11 = ... + M7.
    # W for M7:
    # C11: k=0, i=0 -> idx 0.
    W_s[0, 6] = 1

    # Kronecker product for 4x4x4
    # U1: 16x49
    U1 = np.zeros((16, 49), dtype=float)
    V1 = np.zeros((20, 49), dtype=float) # p=5, so 4*5=20 rows
    W1 = np.zeros((20, 49), dtype=float) # n=4, so 5*4=20 rows

    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                if U_s[i1*2+j1, c1] != 0:
                    for i2 in range(2):
                        for j2 in range(2):
                            for c2 in range(7):
                                if U_s[i2*2+j2, c2] != 0:
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    c = c1*7 + c2
                                    U1[i*4+j, c] += U_s[i1*2+j1, c1] * U_s[i2*2+j2, c2]
    
    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(7):
                if V_s[j1*2+k1, c1] != 0:
                    for j2 in range(2):
                        for k2 in range(2):
                            for c2 in range(7):
                                if V_s[j2*2+k2, c2] != 0:
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    c = c1*7 + c2
                                    # V row index: j*p + k. p=5.
                                    V1[j*5+k, c] += V_s[j1*2+k1, c1] * V_s[j2*2+k2, c2]

    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(7):
                if W_s[k1*2+i1, c1] != 0:
                    for k2 in range(2):
                        for i2 in range(2):
                            for c2 in range(7):
                                if W_s[k2*2+i2, c2] != 0:
                                    k = k1*2 + k2
                                    i = i1*2 + i2
                                    c = c1*7 + c2
                                    # W row index: k*n + i. n=4.
                                    W1[k*4+i, c] += W_s[k1*2+i1, c1] * W_s[k2*2+i2, c2]

    # Part 2: 4x4x1 Schoolbook (Rank 16)
    U2 = np.zeros((16, 16), dtype=float)
    V2 = np.zeros((20, 16), dtype=float)
    W2 = np.zeros((20, 16), dtype=float)
    
    for i in range(4):
        for j in range(4):
            c = i*4 + j
            U2[i*4+j, c] = 1
            # V row: j*5 + 4 (k=4)
            V2[j*5+4, c] = 1
            # W row: 4*4 + i (k=4, i)
            W2[4*4+i, c] = 1

    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Convert to list of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
