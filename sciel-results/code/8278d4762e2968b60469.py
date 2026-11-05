import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_rank1(U_vec, V_vec, W_vec):
        U_parts.append(U_vec)
        V_parts.append(V_vec)
        W_parts.append(W_vec)
    
    # Block 1: C[0:2, 0:2] = A[0:2, :] * B[:, 0:2] (2x4x2, rank 8)
    for i in range(2):
        for j in range(4):
            for k in range(2):
                U_vec = np.zeros(n * m)
                V_vec = np.zeros(m * p)
                W_vec = np.zeros(p * n)
                U_vec[i * m + j] = 1
                V_vec[j * p + k] = 1
                W_vec[k * n + i] = 1
                add_rank1(U_vec, V_vec, W_vec)
    
    # Block 2: C[0:2, 2:4] = A[0:2, :] * B[:, 2:4] (2x4x2, rank 8)
    for i in range(2):
        for j in range(4):
            for k in range(2, 4):
                U_vec = np.zeros(n * m)
                V_vec = np.zeros(m * p)
                W_vec = np.zeros(p * n)
                U_vec[i * m + j] = 1
                V_vec[j * p + k] = 1
                W_vec[k * n + i] = 1
                add_rank1(U_vec, V_vec, W_vec)
    
    # Block 3: C[0:2, 4:5] = A[0:2, :] * B[:, 4:5] (2x4x1, rank 8)
    for i in range(2):
        for j in range(4):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            U_vec[i * m + j] = 1
            V_vec[j * p + 4] = 1
            W_vec[4 * n + i] = 1
            add_rank1(U_vec, V_vec, W_vec)
    
    # Block 4: C[2:3, 0:2] = A[2:3, :] * B[:, 0:2] (1x4x2, rank 8)
    for j in range(4):
        for k in range(2):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            U_vec[2 * m + j] = 1
            V_vec[j * p + k] = 1
            W_vec[k * n + 2] = 1
            add_rank1(U_vec, V_vec, W_vec)
    
    # Block 5: C[2:3, 2:4] = A[2:3, :] * B[:, 2:4] (1x4x2, rank 8)
    for j in range(4):
        for k in range(2, 4):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            U_vec[2 * m + j] = 1
            V_vec[j * p + k] = 1
            W_vec[k * n + 2] = 1
            add_rank1(U_vec, V_vec, W_vec)
    
    # Block 6: C[2:3, 4:5] = A[2:3, :] * B[:, 4:5] (1x4x1, rank 4)
    for j in range(4):
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        U_vec[2 * m + j] = 1
        V_vec[j * p + 4] = 1
        W_vec[4 * n + 2] = 1
        add_rank1(U_vec, V_vec, W_vec)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
