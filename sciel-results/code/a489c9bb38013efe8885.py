import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 solution using 4 Strassen blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen factors for 2x2x2
    # Indices: U[i*2+k], V[k*2+j], W[j*2+i]
    # r=0: M1
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 1, 0]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [0, 1, -1, 0, 0, 1, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 1, -1, 1, 1]
    ])
    
    # Blocks:
    # 1. k in {0,1}, j in {0,1}
    # 2. k in {0,1}, j in {2,3}
    # 3. k in {2,3}, j in {0,1}
    # 4. k in {2,3}, j in {2,3}
    
    col_offset = 0
    for k_start in [0, 2]:
        for j_start in [0, 2]:
            # Local indices k' in {0,1}, j' in {0,1}
            for r in range(7):
                # U: i in {0,1}, k' in {0,1} -> global k = k' + k_start
                for i in range(2):
                    for k_prime in range(2):
                        global_k = k_prime + k_start
                        U[i * m + global_k, col_offset + r] += U_s[i * 2 + k_prime, r]
                        
                # V: k' in {0,1}, j' in {0,1} -> global k = k' + k_start, global j = j' + j_start
                for k_prime in range(2):
                    for j_prime in range(2):
                        global_k = k_prime + k_start
                        global_j = j_prime + j_start
                        V[global_k * p + global_j, col_offset + r] += V_s[k_prime * 2 + j_prime, r]
                        
                # W: j' in {0,1}, i in {0,1} -> global j = j' + j_start
                for j_prime in range(2):
                    for i in range(2):
                        global_j = j_prime + j_start
                        W[global_j * n + i, col_offset + r] += W_s[j_prime * 2 + i, r]
                        
            col_offset += 7
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
