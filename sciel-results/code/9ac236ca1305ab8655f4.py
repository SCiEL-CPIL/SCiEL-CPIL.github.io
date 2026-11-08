"""Rewritten program for matrix multiplication 3,4,8 using Kronecker product decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 98
    # Part 1: 28 terms for the 1x4x8 block (i=0)
    # Part 2: 70 terms for the 2x4x8 block (i=1,2) using 10 Strassen 2x2x2 blocks
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7
    # Schoolbook for 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    
    # Strassen local factors
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1], # M3: C01+=1, C11+=1
        [1, 0, 1, 0], # M4: C00+=1, C10+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, 1], # M6: C11+=1
        [1, 0, 0, 0]  # M7: C00+=1
    ]
    
    # Iterate over j_blocks and k_blocks
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            # 7 Strassen terms
            for t in range(7):
                # Map local indices to global
                # Local A index l -> global i, j
                # l=0 -> i'=0, j'=0 -> global i=1, j=j_start+0
                # l=1 -> i'=0, j'=1 -> global i=1, j=j_start+1
                # l=2 -> i'=1, j'=0 -> global i=2, j=j_start+0
                # l=3 -> i'=1, j'=1 -> global i=2, j=j_start+1
                
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                        
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                        
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
