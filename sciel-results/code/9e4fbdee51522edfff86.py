import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct the 6x6 decomposition as the Kronecker product of
    # a 2x2 Strassen decomposition (rank 7) and a 3x3 schoolbook decomposition (rank 27).
    # Total rank = 7 * 27 = 189.
    
    # 1. Define Strassen 2x2 decomposition
    # U2, V2, W2 are 4x7 matrices
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen terms
    # m1: (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # m2: (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C_01
    
    # m3: a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C_10
    
    # m4: d(g-h)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[3, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1 # C_01, C_11
    
    # m5: (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C_00, C_10
    
    # m6: (c-d)(e+f)
    U2[2, 5] = 1; U2[3, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = 1 # C_00
    
    # m7: (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[3, 6] = 1 # C_11
    
    # 2. Define Schoolbook 3x3 decomposition
    # U3, V3, W3 are 9x27 matrices
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1
                
    # 3. Kronecker Product
    # U = U2 \otimes U3 ?
    # The tensor for 6x6 is T6 = T2 \otimes T3.
    # U6 = U2 \otimes U3 (shape 4*9=36, 7*27=189)
    # V6 = V2 \otimes V3
    # W6 = W2 \otimes W3
    
    U6 = np.kron(U2, U3)
    V6 = np.kron(V2, V3)
    W6 = np.kron(W2, W3)
    
    # Convert to lists
    U = U6.tolist()
    V = V6.tolist()
    W = W6.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
