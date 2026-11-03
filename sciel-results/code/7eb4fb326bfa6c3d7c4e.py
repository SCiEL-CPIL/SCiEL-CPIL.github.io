import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank 72 decomposition
    # Part 1: i=0 (1x4x8) -> 8 blocks of 1x2x2 (rank 2 each) -> 16 terms
    # Part 2: i=1,2 (2x4x8) -> 8 blocks of 2x2x2 (rank 7 each, Strassen) -> 56 terms
    # Total rank = 72
    
    rank = 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0
    # j blocks: 0, 1 (size 2)
    # k blocks: 0, 1, 2, 3 (size 2)
    for j_block in range(2):
        for k_block in range(4):
            j_base = j_block * 2
            k_base = k_block * 2
            
            # Term 1: j=0
            j = j_base
            for k in range(2):
                k_g = k_base + k
                U[0 * m + j, col] = 1.0
                V[j * p + k_g, col] = 1.0
                W[k_g * n + 0, col] = 1.0
                col += 1
                
            # Term 2: j=1
            j = j_base + 1
            for k in range(2):
                k_g = k_base + k
                U[0 * m + j, col] = 1.0
                V[j * p + k_g, col] = 1.0
                W[k_g * n + 0, col] = 1.0
                col += 1

    # Part 2: i=1,2
    # Strassen 2x2x2
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    for j_block in range(2):
        for k_block in range(4):
            j_base = j_block * 2
            k_base = k_block * 2
            for t in range(7):
                # U
                for l in range(4):
                    if S_A[t, l] != 0:
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_base + loc_j
                        U[g_i * m + g_j, col] += S_A[t, l]
                        
                # V
                for l in range(4):
                    if S_B[t, l] != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_base + loc_j
                        g_k = k_base + loc_k
                        V[g_j * p + g_k, col] += S_B[t, l]
                        
                # W
                for l in range(4):
                    if S_C[t, l] != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_base + loc_k
                        W[g_k * n + g_i, col] += S_C[t, l]
                        
                col += 1
                
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
