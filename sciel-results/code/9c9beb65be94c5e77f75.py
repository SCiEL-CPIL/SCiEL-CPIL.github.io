import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, 1x4x8 schoolbook (32 terms)
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2, 2x4x8 Strassen (56 terms)
    # Strassen local factors for 2x2x2
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
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
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    # Iterate over j_blocks (2) and k_blocks (4)
    # j_blocks cover the 4 columns of A (and rows of B)
    # k_blocks cover the 8 columns of B (and rows of C)
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                # Fill U: A factors
                # l=0 -> local (0,0) -> global i=1, j=j_start
                # l=1 -> local (0,1) -> global i=1, j=j_start+1
                # l=2 -> local (1,0) -> global i=2, j=j_start
                # l=3 -> local (1,1) -> global i=2, j=j_start+1
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                        
                # Fill V: B factors
                # l=0 -> local (0,0) -> global j=j_start, k=k_start
                # l=1 -> local (0,1) -> global j=j_start, k=k_start+1
                # l=2 -> local (1,0) -> global j=j_start+1, k=k_start
                # l=3 -> local (1,1) -> global j=j_start+1, k=k_start+1
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                        
                # Fill W: C factors
                # l=0 -> local (0,0) -> global i=1, k=k_start
                # l=1 -> local (0,1) -> global i=1, k=k_start+1
                # l=2 -> local (1,0) -> global i=2, k=k_start
                # l=3 -> local (1,1) -> global i=2, k=k_start+1
                # W index is k*n + i
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
