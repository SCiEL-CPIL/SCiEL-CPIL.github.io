import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = 53
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen components for 2x2x2
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook components for 2x2x1
    U_b = np.eye(4, 4)
    V_b = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_b = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Assembly
    col = 0
    for block_idx in range(7):
        k_start = block_idx * 2
        
        U[:, col:col+7] = U_s
        
        for j in range(2):
            for k_rel in range(2):
                v_row = j * p + (k_start + k_rel)
                v_s_row = j * 2 + k_rel
                V[v_row, col:col+7] = V_s[v_s_row, :]
                
        for k_rel in range(2):
            for i in range(2):
                w_row = (k_start + k_rel) * n + i
                w_s_row = k_rel * 2 + i
                W[w_row, col:col+7] = W_s[w_s_row, :]
                
        col += 7
        
    k_start = 14
    U[:, col:col+4] = U_b
    
    for j in range(2):
        v_row = j * p + 14
        V[v_row, col:col+4] = V_b[j, :]
        
    for i in range(2):
        w_row = 14 * n + i
        W[w_row, col:col+4] = W_b[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
