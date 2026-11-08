"""Rewritten program for matrix multiplication 3,4,8 using block-partitioning with Strassen."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Part 1: 32 terms for the 1x4x8 block (i=0) - schoolbook
    # Part 2: 56 terms for the 2x4x8 block (i=1,2) using 8 Strassen 2x2x2 blocks
    # Total rank: 32 + 56 = 88
    
    rank = 88
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
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Iterate over j_blocks and k_blocks
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            # 7 Strassen terms
            for t in range(7):
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
