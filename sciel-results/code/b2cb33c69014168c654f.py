import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, 1x4x8 schoolbook
    for j in range(4):
        for k in range(8):
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
    # Part 2: i=1,2, 2x4x8 Strassen
    # Strassen local factors for 2x2x2
    strassen_U = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ])
    strassen_V = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    strassen_W = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ])
    
    # Iterate over j_blocks (2) and k_blocks (4)
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Update U, V, W for this Strassen term
                for l in range(4):
                    val = strassen_U[t, l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j, col] += val
                        
                    val = strassen_V[t, l]
                    if val != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k, col] += val
                        
                    val = strassen_W[t, l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i, col] += val
                        
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
