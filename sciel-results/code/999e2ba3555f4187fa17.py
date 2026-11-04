import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target: 3x3 * 3x5 -> 3x5
    # Rank 44 decomposition
    
    rank = 44
    U = np.zeros((n*m, rank)) # 9x44
    V = np.zeros((m*p, rank)) # 15x44
    W = np.zeros((p*n, rank)) # 15x44
    
    col = 0
    
    # Helper to add a block decomposition
    # A_block: (na x ma), B_block: (mb x pb), C_block: (na x pb)
    # Indices in global U, V, W
    
    # 1. 2x2x2 (Strassen) for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen 7 terms
    # A indices: 0,1,2,3 (flattened 2x2)
    # B indices: 0,1,2,3 (flattened 2x2)
    # C indices: 0,1,2,3 (flattened 2x2)
    
    # Strassen coefficients
    # M1 = (a11+a22)(b11+b22)
    # M2 = a22(b21-b11)
    # M3 = (a11+a12)b22
    # M4 = a22(b12-b21)
    # M5 = (a21+a22)b11
    # M6 = (a11-a21)(b11+b21)
    # M7 = (a12-a22)(b12+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M6
    # C21 = M2 + M5
    # C22 = M1 - M3 + M6 + M4
    
    strassen_U = [
        [1, 0, 1, 0], # a11+a22
        [0, 0, 0, 1], # a22
        [1, 1, 0, 0], # a11+a12
        [0, 0, 0, 1], # a22
        [0, 1, 0, 1], # a21+a22
        [1, 0, -1, 0], # a11-a21
        [0, 1, 0, -1]  # a12-a22
    ]
    strassen_V = [
        [1, 1, 0, 0], # b11+b22
        [0, 0, 1, -1], # b21-b11
        [0, 0, 0, 1], # b22
        [1, -1, 0, 0], # b12-b21
        [1, 0, 0, 0], # b11
        [1, 1, 0, 0], # b11+b21
        [0, 0, 1, 1]  # b12+b22
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1 + M4 - M5 + M7 -> C11
        [0, 1, 0, 0], # M3 + M6 -> C12
        [0, 0, 1, 0], # M2 + M5 -> C21
        [1, 0, 0, 1]  # M1 - M3 + M6 + M4 -> C22
    ]
    # Wait, W coefficients need to match the output indices.
    # C11 is index 0, C12 is index 1, C21 is index 2, C22 is index 3.
    # W is (p*n, rank). Here p=2, n=2.
    # W[gamma, r] = coeff.
    
    for r in range(7):
        # U
        for i in range(4):
            if strassen_U[r][i] != 0:
                U[i, col] = strassen_U[r][i]
        # V
        for j in range(4):
            if strassen_V[r][j] != 0:
                V[j, col] = strassen_V[r][j]
        # W
        # C11 (0): M1(1) + M4(1) - M5(1) + M7(1)
        # C12 (1): M3(1) + M6(1)
        # C21 (2): M2(1) + M5(1)
        # C22 (3): M1(1) - M3(1) + M6(1) + M4(1)
        
        # M1: C11(1), C22(1)
        if r == 0:
            W[0, col] = 1
            W[3, col] = 1
        # M2: C21(1)
        elif r == 1:
            W[2, col] = 1
        # M3: C12(1), C22(-1)
        elif r == 2:
            W[1, col] = 1
            W[3, col] = -1
        # M4: C11(1), C22(1)
        elif r == 3:
            W[0, col] = 1
            W[3, col] = 1
        # M5: C11(-1), C21(1)
        elif r == 4:
            W[0, col] = -1
            W[2, col] = 1
        # M6: C12(1), C22(1)
        elif r == 5:
            W[1, col] = 1
            W[3, col] = 1
        # M7: C11(1)
        elif r == 6:
            W[0, col] = 1
            
        col += 1
        
    # 2. 2x2x3 (Schoolbook) for A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # A indices 0-3, B indices 4-9 (2x3), C indices 2-7 (2x3)
    # B global indices: B[0, 2:5] are indices 2,3,4. B[1, 2:5] are indices 7,8,9.
    # C global indices: C[0, 2:5] are indices 2,3,4. C[1, 2:5] are indices 7,8,9.
    
    for i in range(2):
        for k in range(2):
            for j in range(3):
                # A[i, k]
                U[i*2+k, col] = 1
                # B[k, 2+j]
                V[k*5 + (2+j), col] = 1
                # C[i, 2+j]
                W[(2+j)*3 + i, col] = 1
                col += 1
                
    # 3. 2x1x5 (Schoolbook) for A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    # A indices 4,5. B indices 10-14. C indices 0-4 (row 0), 5-9 (row 1).
    
    for i in range(2):
        for j in range(5):
            U[2*2+i, col] = 1 # A[i, 2]
            V[2*5+j, col] = 1 # B[2, j]
            W[j*3+i, col] = 1 # C[i, j]
            col += 1
            
    # 4. 1x2x5 (Schoolbook) for A[2:3, 0:2] * B[0:2, 0:5] -> C[2:3, 0:5]
    # A indices 6,7. B indices 0-4 (row 0), 5-9 (row 1). C indices 10-14 (row 2).
    
    for k in range(2):
        for j in range(5):
            U[2*3+k, col] = 1 # A[2, k]
            V[k*5+j, col] = 1 # B[k, j]
            W[j*3+2, col] = 1 # C[2, j]
            col += 1
            
    # 5. 1x1x5 (Schoolbook) for A[2:3, 2:3] * B[2:3, 0:5] -> C[2:3, 0:5]
    # A index 8. B indices 10-14. C indices 10-14.
    
    for j in range(5):
        U[2*3+2, col] = 1 # A[2, 2]
        V[2*5+j, col] = 1 # B[2, j]
        W[j*3+2, col] = 1 # C[2, j]
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
