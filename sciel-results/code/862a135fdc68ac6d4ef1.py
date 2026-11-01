import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen rank-7 factors for 2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: i,j in {0,1} for U, j,k in {0,1} for V, k,i in {0,1} for W
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen terms:
    # 1: (A00+A11)(B00+B11)
    # 2: (A10+A11)B00
    # 3: A00(B10-B11)
    # 4: A11(B01-B00)
    # 5: (A00+A01)B11
    # 6: (A10-A00)(B00+B01)
    # 7: (A01-A11)(B10+B11)
    
    # U2[i,j, r]
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    U2[0, 0, 2] = 1
    U2[1, 1, 3] = 1
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    
    # V2[j,k, r]
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    V2[0, 0, 1] = 1
    V2[1, 0, 2] = 1; V2[1, 1, 2] = -1
    V2[0, 1, 3] = 1; V2[0, 0, 3] = -1
    V2[1, 1, 4] = 1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    
    # W2[k,i, r]
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    W2[0, 1, 1] = 1
    W2[0, 0, 2] = 1; W2[1, 0, 2] = -1
    W2[1, 1, 3] = 1; W2[0, 1, 3] = -1
    W2[1, 1, 4] = 1
    W2[0, 0, 5] = 1; W2[1, 0, 5] = 1
    W2[1, 1, 6] = 1; W2[0, 1, 6] = -1
    
    # Construct 4x4 factors via Kronecker product
    # Rank 49
    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r2]
                            
                            k1 = j1 # No, k is separate
                            # V indices are j,k
                            for k1 in range(2):
                                for k0 in range(2):
                                    k = k1 * 2 + k0
                                    V4[j * 4 + k, r] = V2[j1 * 2 + k1, r1] * V2[j0 * 2 + k0, r2]
                                    
                                    # W indices are k,i
                                    W4[k * 4 + i, r] = W2[k1 * 2 + i1, r1] * W2[k0 * 2 + i0, r2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END
