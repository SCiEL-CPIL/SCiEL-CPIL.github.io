import numpy as np
import time
import random

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[2, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 140
    
    # Baseline Rank 53
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    baseline_rank = 53
    U = np.zeros((n * m, baseline_rank))
    V = np.zeros((m * p, baseline_rank))
    W = np.zeros((p * n, baseline_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
