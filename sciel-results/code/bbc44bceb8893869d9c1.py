import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 1, -1],
        [0, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ])
    
    # Kronecker product to get 4x4x4 rank 49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    rank_4x4 = 49
    rank_total = rank_4x4 * 2
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Part 1: C1 = A * B1
    U[:, 0:rank_4x4] = U4
    V[0:m*4, 0:rank_4x4] = V4
    W[0:4*n, 0:rank_4x4] = W4
    
    # Part 2: C2 = A * B2
    U[:, rank_4x4:rank_total] = U4
    V[m*4:m*p, rank_4x4:rank_total] = V4
    W[4*n:8*n, rank_4x4:rank_total] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
