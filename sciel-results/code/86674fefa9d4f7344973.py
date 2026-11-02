import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Strassen's algorithm for 2x2 matrices: rank 7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, -1, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, -1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0]
    ])
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    
    # 4x4 Strassen via Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 4x1 trivial decomposition
    U41 = np.eye(16)
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            V41[j, i*4+j] = 1
            W41[i, i*4+j] = 1
    
    rank_total = 49 + 49 + 16  # 114
    
    U = np.zeros((16, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    # Block 1: B[:, 0:4], C[:, 0:4]
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Block 2: B[:, 4:8], C[:, 4:8]
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Block 3: B[:, 8], C[:, 8]
    U[:, 98:114] = U41
    V[32:36, 98:114] = V41
    W[32:36, 98:114] = W41
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
