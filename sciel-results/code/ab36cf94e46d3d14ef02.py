import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # i,j,k in {0,1}
    
    # Term 1: (a00+a11)(b00+b11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # Term 2: (a10+a11)b00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1; W1[3, 1] = -1
    
    # Term 3: a00(b01-b11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # Term 4: a11(b10-b00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # Term 5: (a00+a01)b11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[1, 4] = 1
    
    # Term 6: (a10-a00)(b00+b01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # Term 7: (a01-a11)(b10+b11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    
    # Schoolbook decomposition for 1x2x3
    # U2: 2x6, V2: 6x6, W2: 3x6
    U2 = np.zeros((2, 6), dtype=float)
    V2 = np.zeros((6, 6), dtype=float)
    W2 = np.zeros((3, 6), dtype=float)
    
    # j' in {0,1}, k' in {0,1,2}
    # U2 row j'
    # V2 row 2*j' + k'
    # W2 row k'
    
    for r in range(6):
        jp = r // 3
        kp = r % 3
        U2[jp, r] = 1
        V2[2*jp + kp, r] = 1
        W2[kp, r] = 1
        
    # Kronecker Products
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Permute V
    V_final = np.zeros_like(V)
    for r in range(42):
        # V kron index K_V
        # K_V = 12*jA + 6*kA + 3*jB + kB
        # We need to iterate over all combinations of jA, kA, jB, kB
        # and map to G_V
        pass
        
    # Better: iterate over all indices
    V_perm = np.zeros_like(V)
    for jA in range(2):
        for kA in range(2):
            for jB in range(2):
                for kB in range(3):
                    K_V = 12*jA + 6*kA + 3*jB + kB
                    G_V = 12*jA + 6*jB + 3*kA + kB
                    V_perm[G_V, :] = V[K_V, :]
    V = V_perm
    
    # Permute W
    W_perm = np.zeros_like(W)
    for kA in range(2):
        for i in range(2):
            for kB in range(3):
                K_W = 6*kA + 3*i + kB
                G_W = 6*kA + 2*kB + i
                W_perm[G_W, :] = W[K_W, :]
    W = W_perm
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
