import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 5
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    rank = 36
    U = np.zeros((dim_u, rank), dtype=int)
    V = np.zeros((dim_v, rank), dtype=int)
    W = np.zeros((dim_w, rank), dtype=int)
    
    # Correct Strassen 2x2x2 matrices
    # U: (4, 7) -> indices i*2+j
    # V: (4, 7) -> indices j*2+k
    # W: (4, 7) -> indices k*2+i
    U_block = np.array([
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0]
    ])
    V_block = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, -1, 0, 1, 0],
        [-1, 0, 1, 0, 0, 0, 1]
    ])
    W_block = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, -1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ])

    col_idx = 0
    # 4 blocks of 2x2x2
    for j_base in [0, 2]:
        for k_base in [0, 2]:
            for r in range(7):
                # U part
                for i in range(2):
                    for j_local in range(2):
                        j = j_base + j_local
                        U[i*m + j, col_idx] = U_block[i*2 + j_local][r]
                
                # V part
                for j_local in range(2):
                    for k_local in range(2):
                        j = j_base + j_local
                        k = k_base + k_local
                        V[j*p + k, col_idx] = V_block[j_local*2 + k_local][r]
                
                # W part
                for k_local in range(2):
                    for i in range(2):
                        k = k_base + k_local
                        W[k*n + i, col_idx] = W_block[k_local*2 + i][r]
                
                col_idx += 1
    
    # 8 terms for k=4 (schoolbook)
    for i in range(n):
        for j in range(m):
            U[i*m + j, col_idx] = 1
            V[j*p + 4, col_idx] = 1
            W[4*n + i, col_idx] = 1
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
