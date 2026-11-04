import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Strassen 2x2 factors
    # U, V, W are lists of 7 vectors of length 4
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 56
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # We partition the m=4 dimension into 2 groups of 2 (g_j in 0,1)
    # We partition the p=8 dimension into 4 groups of 2 (g_k in 0..3)
    # The n=2 dimension is not partitioned further.
    # Each sub-problem is 2x2 * 2x2 -> 2x2
    # Total sub-problems: 2 * 4 = 8
    # Each uses 7 Strassen multiplications.
    # Total rank: 8 * 7 = 56
    
    for g_k in range(4):
        for g_j in range(2):
            for r_prime in range(7):
                u_vals = strassen_U[r_prime]
                v_vals = strassen_V[r_prime]
                w_vals = strassen_W[r_prime]
                
                # U indices: i (0,1) and j (2*g_j .. 2*g_j+1)
                # u_idx_global = i * m + j_global
                for i in range(2):
                    for j_sub in range(2):
                        j_global = 2 * g_j + j_sub
                        u_idx_global = i * m + j_global
                        U[u_idx_global][col] = u_vals[i * 2 + j_sub]
                        
                # V indices: j (2*g_j .. 2*g_j+1) and k (2*g_k .. 2*g_k+1)
                # v_idx_global = j_global * p + k_global
                for j_sub in range(2):
                    j_global = 2 * g_j + j_sub
                    for k_sub in range(2):
                        k_global = 2 * g_k + k_sub
                        v_idx_global = j_global * p + k_global
                        V[v_idx_global][col] = v_vals[j_sub * 2 + k_sub]
                        
                # W indices: k (2*g_k .. 2*g_k+1) and i (0,1)
                # w_idx_global = k_global * n + i
                for k_sub in range(2):
                    k_global = 2 * g_k + k_sub
                    for i in range(2):
                        w_idx_global = k_global * n + i
                        W[w_idx_global][col] = w_vals[k_sub * 2 + i]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
