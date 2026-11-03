import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Define 2x2 Strassen matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation to map Kronecker order to Standard Row-Major order
    # S: Standard Index 0..15
    # R = S // 4, C = S % 4
    # r_in = R % 2, r_out = R // 2
    # c_in = C % 2, c_out = C // 2
    # K_in = r_in * 2 + c_in
    # K_out = r_out * 2 + c_out
    # K = K_out * 4 + K_in
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        R = S // 4
        C = S % 4
        r_in = R % 2
        r_out = R // 2
        c_in = C % 2
        c_out = C // 2
        K_in = r_in * 2 + c_in
        K_out = r_out * 2 + c_out
        K = K_out * 4 + K_in
        perm[S] = K
        
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
