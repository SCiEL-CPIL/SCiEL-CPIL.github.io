import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = 2, 3, 6
    rank = 6
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct the decomposition
    # Rank r = 2*j + s, where j in {0, 1, 2} and s in {0, 1}
    for j in range(m):
        r0 = 2 * j
        r1 = 2 * j + 1
        
        # U: rows i*m + j for i in {0, 1}
        # U[i*m + j, r0] = 1 if i == 0 else 0
        # U[i*m + j, r1] = 1 if i == 1 else 0
        U[0 * m + j, r0] = 1.0
        U[0 * m + j, r1] = 0.0
        U[1 * m + j, r0] = 0.0
        U[1 * m + j, r1] = 1.0
        
        # V: rows j*p + k for k in {0..5}
        # V[j*p + k, r0] = 1
        # V[j*p + k, r1] = 1
        for k in range(p):
            V[j * p + k, r0] = 1.0
            V[j * p + k, r1] = 1.0
            
        # W: rows k*n + i for k in {0..5}, i in {0, 1}
        # W[k*n + 0, r0] = 1
        # W[k*n + 0, r1] = 0
        # W[k*n + 1, r0] = 0
        # W[k*n + 1, r1] = 1
        for k in range(p):
            W[k * n + 0, r0] = 1.0
            W[k * n + 0, r1] = 0.0
            W[k * n + 1, r0] = 0.0
            W[k * n + 1, r1] = 1.0
            
    # Verify
    T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
                
    if not np.allclose(T_check, T_target, atol=1e-9):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
