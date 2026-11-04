import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        i1, i2 = i // 2, i % 2
        j1, j2 = j // 2, j % 2
        for t in range(7):
            for s in range(7):
                c = t * 7 + s
                U4[idx, c] = U_2[i1 * 2 + j1, t] * U_2[i2 * 2 + j2, s]
    
    for idx in range(16):
        j = idx // 4
        k = idx % 4
        j1, j2 = j // 2, j % 2
        k1, k2 = k // 2, k % 2
        for t in range(7):
            for s in range(7):
                c = t * 7 + s
                V4[idx, c] = V_2[j1 * 2 + k1, t] * V_2[j2 * 2 + k2, s]
    
    for idx in range(16):
        k = idx // 4
        i = idx % 4
        k1, k2 = k // 2, k % 2
        i1, i2 = i // 2, i % 2
        for t in range(7):
            for s in range(7):
                c = t * 7 + s
                W4[idx, c] = W_2[k1 * 2 + i1, t] * W_2[k2 * 2 + i2, s]
    
    U[:] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :] = V4[j * 4 + k, :]
            V[j * 8 + (k + 4), :] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :] = W4[k * 4 + i, :]
            W[(k + 4) * 4 + i, :] = W4[k * 4 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
