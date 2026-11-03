import numpy as np

def construct():
    n, m, p = 3, 3, 8
    # Split B into B1 (3 cols), B2 (3 cols), B3 (2 cols)
    # C = A B1 + A B2 + A B3
    
    # U, V, W for M_{3,3,3} (schoolbook)
    # U shape (9, 9), V shape (9, 9), W shape (9, 9)
    # Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For M_{3,3,3}: n=3, m=3, p=3
    U1 = np.zeros((9, 9))
    V1 = np.zeros((9, 9))
    W1 = np.zeros((9, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U1[i*3+j, col] = 1
                V1[j*3+k, col] = 1
                W1[k*3+i, col] = 1
                col += 1
                
    # For M_{3,3,2}: n=3, m=3, p=2
    U2 = np.zeros((9, 18))
    V2 = np.zeros((9, 18)) # m*p = 3*2 = 6? No, V is (m*p, rank). m=3, p=2. Size 6.
    # Wait, V dimension is m*p. For B1 (3x3), V is 9x9.
    # For B3 (3x2), V is 6x18.
    V2 = np.zeros((6, 18))
    W2 = np.zeros((6, 18)) # p*n = 2*3 = 6.
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U2[i*3+j, col] = 1
                V2[j*2+k, col] = 1
                W2[k*3+i, col] = 1
                col += 1

    # Combine
    # Total rank = 9 + 9 + 18 = 36
    rank = 36
    U = np.zeros((9, rank))
    V = np.zeros((24, rank)) # m*p = 3*8 = 24
    W = np.zeros((24, rank)) # p*n = 8*3 = 24
    
    # Place B1 (cols 0-2)
    # V1 corresponds to B1 (j*3+k, k=0..2)
    # In global V, B1 is at rows j*8+k for k=0..2
    for c in range(9):
        U[:, c] = U1[:, c]
        for j in range(3):
            for k in range(3):
                V[j*8+k, c] = V1[j*3+k, c]
        for i in range(3):
            for k in range(3):
                W[k*3+i, c] = W1[k*3+i, c]
                
    # Place B2 (cols 3-5)
    # V1 corresponds to B2 (j*3+k, k=0..2)
    # In global V, B2 is at rows j*8+k for k=3..5
    for c in range(9):
        U[:, 9+c] = U1[:, c]
        for j in range(3):
            for k in range(3):
                V[j*8+(3+k), 9+c] = V1[j*3+k, c]
        for i in range(3):
            for k in range(3):
                W[(3+k)*3+i, 9+c] = W1[k*3+i, c]
                
    # Place B3 (cols 6-7)
    # V2 corresponds to B3 (j*2+k, k=0..1)
    # In global V, B3 is at rows j*8+k for k=6..7
    for c in range(18):
        U[:, 18+c] = U2[:, c]
        for j in range(3):
            for k in range(2):
                V[j*8+(6+k), 18+c] = V2[j*2+k, c]
        for i in range(3):
            for k in range(2):
                W[(6+k)*3+i, 18+c] = W2[k*3+i, c]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
