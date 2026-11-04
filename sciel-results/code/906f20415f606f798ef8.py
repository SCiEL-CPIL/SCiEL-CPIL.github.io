import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U vectors (A), V vectors (B), W vectors (C)
    # Flattened 2x2 matrices, row-major: [00, 01, 10, 11]
    
    U2 = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    V2 = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    W2 = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11
        [0, 1, 0, 1],  # M3: C01 + C11
        [1, 0, 1, 0],  # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # Get 2x2 matrices
            u1 = np.array(U2[r1]).reshape(2, 2)
            u2 = np.array(U2[r2]).reshape(2, 2)
            v1 = np.array(V2[r1]).reshape(2, 2)
            v2 = np.array(V2[r2]).reshape(2, 2)
            w1 = np.array(W2[r1]).reshape(2, 2)
            w2 = np.array(W2[r2]).reshape(2, 2)
            
            # Kronecker product
            u4 = np.kron(u1, u2)
            v4 = np.kron(v1, v2)
            w4 = np.kron(w1, w2)
            
            # Flatten and populate
            for idx in range(16):
                U[idx][col] = int(u4.flat[idx])
                V[idx][col] = int(v4.flat[idx])
                W[idx][col] = int(w4.flat[idx])
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
