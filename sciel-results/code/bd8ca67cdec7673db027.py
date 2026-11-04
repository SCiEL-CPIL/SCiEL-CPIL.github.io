import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 matrices
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_222 = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, -1, 0, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # 6 blocks
    for b in range(6):
        b_row = b // 3  # 0 or 1
        b_col = b % 3   # 0, 1, 2
        
        r_offset = b * 7
        
        for r in range(7):
            r_idx = r_offset + r
            
            # Fill U
            for i in range(2):
                for j_local in range(2):
                    j_global = 2 * b_row + j_local
                    u_idx = i * m + j_global
                    U[u_idx, r_idx] = U_222[i * 2 + j_local, r]
            
            # Fill V
            for j_local in range(2):
                for k_local in range(2):
                    j_global = 2 * b_row + j_local
                    k_global = 2 * b_col + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, r_idx] = V_222[j_local * 2 + k_local, r]
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * b_col + k_local
                    w_idx = k_global * n + i
                    W[w_idx, r_idx] = W_222[k_local * 2 + i, r]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
