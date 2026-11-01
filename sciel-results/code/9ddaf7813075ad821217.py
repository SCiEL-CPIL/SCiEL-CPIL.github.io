"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    
    # Rank 39 decomposition:
    # 5 blocks of 2 columns (Strassen rank 7 each) -> 35
    # 1 block of 1 column (Schoolbook rank 4) -> 4
    total_rank = 39
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen 2x2x2 coefficients
    # U rows: 0:a00, 1:a01, 2:a10, 3:a11
    # V rows: 0:b00, 1:b01, 2:b10, 3:b11
    # W rows: 0:C00, 1:C10, 2:C01, 3:C11 (index k*n+i)
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    col = 0
    # 5 blocks of size 2
    for block_idx in range(5):
        k_start = block_idx * 2
        
        # U is same for all blocks (only depends on i, j)
        U[:, col:col+7] = U_strassen
        
        # V mapping: local (j, k') to global (j, k_start + k')
        for j in range(m):
            for k_local in range(2):
                global_k = k_start + k_local
                v_row_global = j * p + global_k
                v_row_local = j * 2 + k_local
                V[v_row_global, col:col+7] = V_strassen[v_row_local, :]
                
        # W mapping: local (k', i) to global (k_start + k', i)
        for k_local in range(2):
            global_k = k_start + k_local
            for i in range(n):
                w_row_global = global_k * n + i
                w_row_local = k_local * n + i
                W[w_row_global, col:col+7] = W_strassen[w_row_local, :]
        
        col += 7
        
    # 1 block of size 1 (k=10)
    k_start = 10
    
    # Schoolbook 2x2x2x1
    # Products: a00*b0, a01*b1, a10*b0, a11*b1
    # U:
    # 0: a00
    # 1: a01
    # 2: a10
    # 3: a11
    U[:, col:col+4] = np.eye(4)
    
    # V:
    # prod 0: j=0
    # prod 1: j=1
    # prod 2: j=0
    # prod 3: j=1
    V[0 * p + k_start, col + 0] = 1
    V[1 * p + k_start, col + 1] = 1
    V[0 * p + k_start, col + 2] = 1
    V[1 * p + k_start, col + 3] = 1
    
    # W:
    # prod 0: i=0 (C0, k) -> row k_start*n + 0
    # prod 1: i=0 (C0, k) -> row k_start*n + 0
    # prod 2: i=1 (C1, k) -> row k_start*n + 1
    # prod 3: i=1 (C1, k) -> row k_start*n + 1
    W[k_start * n + 0, col + 0] = 1
    W[k_start * n + 0, col + 1] = 1
    W[k_start * n + 1, col + 2] = 1
    W[k_start * n + 1, col + 3] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
