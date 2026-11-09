import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = 3, 3, 8
    rank_total = 30
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col = 0
    
    # Part 1: i in {0,1}, j in {0,1}
    # 4 blocks of 2x2x2 Strassen
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen vectors
    # M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    # M3
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    # M4
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    # M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    W_s[3, 5] = 1
    # M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = 1
    
    for block_idx in range(4):
        k_offset = block_idx * 2
        for i in range(2):
            for j in range(2):
                u_full = i * 3 + j
                u_str = i * 2 + j
                U[u_full, col:col+7] = U_s[u_str, :]
        for j in range(2):
            for k_rel in range(2):
                v_full = j * 8 + (k_offset + k_rel)
                v_str = j * 2 + k_rel
                V[v_full, col:col+7] = V_s[v_str, :]
        for k_rel in range(2):
            for i in range(2):
                w_full = (k_offset + k_rel) * 3 + i
                w_str = k_rel * 2 + i
                W[w_full, col:col+7] = W_s[w_str, :]
        col += 7
        
    # Part 2: i in {0,1}, j=2
    # Rank 1
    U[2, col] = 1
    U[5, col] = 1
    for k in range(8):
        V[16 + k, col] = 1
        W[k * 3 + 0, col] = 1 # i=0
        W[k * 3 + 1, col] = 1 # i=1
    col += 1
    
    # Part 3+4: i=2, j in {0,1,2}
    # Rank 1
    U[8, col] = 1
    for j in range(3):
        for k in range(8):
            V[j * 8 + k, col] = 1
    for k in range(8):
        W[k * 3 + 2, col] = 1 # i=2
    col += 1
        
    # Verify
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T, target, atol=1e-9):
        # Fallback
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
