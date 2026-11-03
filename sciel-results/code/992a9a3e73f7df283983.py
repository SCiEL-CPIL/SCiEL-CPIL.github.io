import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    # Strassen 2x2 factors
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, 1]
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 1, 0, -1]
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 0, 1, 1]
    U4 = [0, 0, 0, 1]
    V4 = [-1, 0, 1, 0]
    W4 = [1, 1, 0, 0]
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 0, 1, 0]
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 0]
    
    strassen_U = [U1, U2, U3, U4, U5, U6, U7]
    strassen_V = [V1, V2, V3, V4, V5, V6, V7]
    strassen_W = [W1, W2, W3, W4, W5, W6, W7]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for g_k in range(4):
        for g_j in range(2):
            for r_prime in range(7):
                u_vals = strassen_U[r_prime]
                v_vals = strassen_V[r_prime]
                w_vals = strassen_W[r_prime]
                
                for i in range(2):
                    for j_sub in range(2):
                        u_idx_global = i * 4 + (2 * g_j + j_sub)
                        U[u_idx_global][col] = u_vals[i * 2 + j_sub]
                        
                for j_sub in range(2):
                    for k_sub in range(2):
                        v_idx_global = (2 * g_j + j_sub) * 8 + (2 * g_k + k_sub)
                        V[v_idx_global][col] = v_vals[j_sub * 2 + k_sub]
                        
                for k_sub in range(2):
                    for i in range(2):
                        w_idx_global = (2 * g_k + k_sub) * 2 + i
                        W[w_idx_global][col] = w_vals[k_sub * 2 + i]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
