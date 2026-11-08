import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # Rank 7
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11 for 2x2 matrix
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0], # 00
        [0, 1, 0, 0, 0, 0, 0], # 01
        [0, 0, 1, 0, 0, 0, 0], # 10
        [0, 0, 0, 1, 0, 0, 0]  # 11
    ])
    # Wait, Strassen is not identity.
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = -1; W2[2, 0] = -1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[0, 2] = 1; W2[2, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1; W2[3, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[1, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1

    # Schoolbook decomposition for 2x2x3
    # Rank 12
    # U3: 4x12, V3: 6x12, W3: 4x12
    # A: 2x2 (4 entries), B: 2x3 (6 entries), C: 2x2 (4 entries)
    U3 = np.zeros((4, 12), dtype=int)
    V3 = np.zeros((6, 12), dtype=int)
    W3 = np.zeros((4, 12), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U3[i*2+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*2+i, col] = 1
                col += 1

    # Kronecker Product
    # U = U2 (Kron) U3
    # V = V2 (Kron) V3
    # W = W2 (Kron) W3
    
    U = np.kron(U2, U3)
    V = np.kron(V2, V3)
    W = np.kron(W2, W3)
    
    rank = U.shape[1]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
