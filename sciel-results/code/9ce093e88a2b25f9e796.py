import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Construct decomposition using Kronecker product of Strassen (2x2x2) and Schoolbook (1x2x3)
    # Strassen Rank 7, Schoolbook Rank 6. Total Rank 42.
    
    # Strassen for 2x2x2
    # U1, V1, W1 for 2x2x2
    # i in 0..3 (2x2), j in 0..3 (2x2), k in 0..3 (2x2)
    # i = i_A * 2 + j_A
    # j = j_B * 2 + k_B
    # k = k_C * 2 + i_C
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # M1 = A00(B00 - B10)
    U1[0, 0] = 1
    V1[0, 0] = 1
    V1[2, 0] = -1
    W1[0, 0] = 1 # C00 += M1? No, C00 = M5 - M4 + M2 - M1. So W1[0,0] = -1
    W1[0, 0] = -1
    W1[1, 0] = 1 # C01 += M1
    
    # M2 = (A00 + A10)B10
    U1[0, 1] = 1
    U1[2, 1] = 1
    V1[2, 1] = 1
    W1[0, 1] = 1 # C00 += M2
    W1[2, 1] = 1 # C10 += M2
    
    # M3 = A11(B01 - B11)
    U1[3, 2] = 1
    V1[1, 2] = 1
    V1[3, 2] = -1
    W1[2, 2] = 1 # C10 += M3
    W1[3, 2] = -1 # C11 -= M3
    
    # M4 = (A11 + A01)B11
    U1[3, 3] = 1
    U1[1, 3] = 1
    V1[3, 3] = 1
    W1[0, 3] = -1 # C00 -= M4
    W1[1, 3] = 1 # C01 += M4
    
    # M5 = (A00 + A11)(B00 + B11)
    U1[0, 4] = 1
    U1[3, 4] = 1
    V1[0, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = 1 # C00 += M5
    W1[3, 4] = 1 # C11 += M5
    
    # M6 = (A01 - A11)(B10 + B11)
    U1[1, 5] = 1
    U1[3, 5] = -1
    V1[2, 5] = 1
    V1[3, 5] = 1
    W1[3, 5] = 1 # C11 += M6
    
    # M7 = (A00 - A10)(B00 + B01)
    U1[0, 6] = 1
    U1[2, 6] = -1
    V1[0, 6] = 1
    V1[1, 6] = 1
    W1[3, 6] = -1 # C11 -= M7
    
    # Schoolbook for 1x2x3
    # n2=1, m2=2, p2=3
    # i2 in 0..1 (1x2), j2 in 0..5 (2x3), k2 in 0..2 (3x1)
    # i2 = i_A2 * 2 + j_A2 (i_A2=0)
    # j2 = j_B2 * 3 + k_B2
    # k2 = k_C2 * 1 + i_C2 (i_C2=0)
    
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    for j in range(2):
        for k in range(3):
            idx = j * 3 + k
            U2[j, idx] = 1
            V2[idx, idx] = 1
            W2[k, idx] = 1
            
    # Kronecker Product
    # U = U1 \otimes U2
    # V = V1 \otimes V2
    # W = W1 \otimes W2
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

