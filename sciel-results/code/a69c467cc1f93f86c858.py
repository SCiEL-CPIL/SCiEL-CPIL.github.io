import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Construct decomposition for 2x2 matrix multiplication using Strassen's algorithm (Rank 7)
    # U, V, W are 4x7 matrices
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen steps:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices: 0=(11), 1=(12), 2=(21), 3=(22) for 2x2 matrix
    # A indices: 0, 1, 2, 3
    # B indices: 0, 1, 2, 3
    # C indices: 0, 1, 2, 3
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C21 += M2, C22 -= M2? 
    # Let's re-verify Strassen formulas
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Re-do W2 based on formulas
    # M1: C11+=1, C22+=1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: C21+=1, C22-=1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: C12+=1, C22+=1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: C11+=1, C21+=1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: C11-=1, C12+=1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: C22+=1
    W2[3, 5] = 1
    
    # M7: C11+=1
    W2[0, 6] = 1
    
    # Verify U2, V2
    # M1: A11+A22, B11+B22
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    
    # M2: A21+A22, B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    
    # M3: A11, B12-B22
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    
    # M4: A22, B21-B11
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    
    # M5: A11+A12, B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    
    # M6: A21-A11, B11+B12
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    
    # M7: A12-A22, B21+B22
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Construct decomposition for 3x3 matrix multiplication using Schoolbook (Rank 27)
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A_ik, B_kj -> C_ij
                U3[i * 3 + k, col] = 1
                V3[k * 3 + j, col] = 1
                W3[i * 3 + j, col] = 1
                col += 1
                
    # Combine 2x2 and 3x3 decompositions to form 6x6 decomposition
    # Rank = 7 * 27 = 189
    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for u3 in range(27):
        for u2 in range(7):
            r = u3 * 7 + u2
            # A part
            for i in range(3):
                for k in range(3):
                    for r2 in range(2):
                        for c2 in range(2):
                            val = U3[i * 3 + k, u3] * U2[r2 * 2 + c2, u2]
                            if val != 0:
                                U[i * 2 + r2, k * 2 + c2, r] = val
            # B part
            for k in range(3):
                for j in range(3):
                    for c2 in range(2):
                        for s2 in range(2):
                            val = V3[k * 3 + j, u3] * V2[c2 * 2 + s2, u2]
                            if val != 0:
                                V[k * 2 + c2, j * 2 + s2, r] = val
            # C part
            for i in range(3):
                for j in range(3):
                    for r2 in range(2):
                        for s2 in range(2):
                            val = W3[i * 3 + j, u3] * W2[r2 * 2 + s2, u2]
                            if val != 0:
                                W[i * 2 + r2, j * 2 + s2, r] = val
                                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
