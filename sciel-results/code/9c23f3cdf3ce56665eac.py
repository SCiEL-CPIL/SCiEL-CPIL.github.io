import numpy as np

def construct():
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    # Permute rows to match row-major flattening
    # R_kron = 4 * r1 + r2
    # r1 = 2 * i1 + j1
    # r2 = 2 * i2 + j2
    # R_actual = 4 * (2 * i1 + i2) + (2 * j1 + j2)
    
    perm = [0] * 16
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    r1 = 2 * i1 + j1
                    r2 = 2 * i2 + j2
                    R_kron = 4 * r1 + r2
                    R_actual = 4 * (2 * i1 + i2) + (2 * j1 + j2)
                    perm[R_kron] = R_actual
    
    U = U[perm, :]
    V = V[perm, :]
    W = W[perm, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
