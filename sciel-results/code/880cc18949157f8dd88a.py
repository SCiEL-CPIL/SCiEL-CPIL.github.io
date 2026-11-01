import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Strassen decomposition for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ])
    
    V_S = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, -1, 1, 0, -1],
        [1, 0, -1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, -1]
    ])
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 1, -1]
    ])
    
    # Trivial decomposition for 1x2x3
    # U_T: 2x6, V_T: 6x6, W_T: 3x6
    U_T = np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1]
    ])
    
    V_T = np.eye(6)
    
    W_T = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ])
    
    # Kronecker products
    U = np.kron(U_S, U_T)
    V = np.kron(V_S, V_T)
    W = np.kron(W_S, W_T)
    
    # Permute W rows to match W[k*n+i, r] convention
    # W is currently in row-major order of C (2x6), i.e., index i*6+k
    # We need index k*2+i
    W_new = np.zeros_like(W)
    for i in range(2):
        for k in range(6):
            W_new[k*2 + i] = W[i*6 + k]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W_new.tolist()}
