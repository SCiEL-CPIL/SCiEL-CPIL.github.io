import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 56 decomposition using 8 blocks of Strassen (rank 7 each)
    n, m, p = DIMENSIONS
    
    # Number of blocks: 2 j-groups, 4 k-groups
    n_j_groups = 2
    n_k_groups = 4
    block_rank = 7
    total_rank = n_j_groups * n_k_groups * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen local factors for 2x2x2
    # U_loc: 4 rows (i*2 + j''), V_loc: 4 rows (j''*2 + k''), W_loc: 4 rows (k''*2 + i)
    # 7 columns
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j''=0
        [0, 0, 0, 0, 0, 0, 1],  # i=0, j''=1
        [0, 1, 1, 0, 1, 0, 0],  # i=1, j''=0
        [1, 1, 0, 1, 0, 1, -1]  # i=1, j''=1
    ]
    
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # j''=0, k''=0
        [0, 0, 1, 0, 0, 1, 0],  # j''=0, k''=1
        [0, 0, 0, 1, 0, 0, 1],  # j''=1, k''=0
        [1, 0, -1, 0, 1, 0, 1]  # j''=1, k''=1
    ]
    
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # k''=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k''=0, i=1
        [0, 0, 0, 0, 0, 0, 0],  # Wait, check W_loc indices
        [0, 0, 1, 0, 0, 1, -1]  # k''=1, i=1
    ]
    
    # Let's re-verify W_loc from previous derivation
    # M1: W[00]=1, W[11]=1 -> k''=0,i=0:1; k''=1,i=1:1
    # M2: W[10]=1 -> k''=0,i=1:1
    # M3: W[01]=1, W[11]=1 -> k''=1,i=0:1; k''=1,i=1:1
    # M4: W[00]=1, W[10]=1 -> k''=0,i=0:1; k''=0,i=1:1
    # M5: W[00]=-1, W[01]=1 -> k''=0,i=0:-1; k''=1,i=0:1
    # M6: W[11]=1 -> k''=1,i=1:1
    # M7: W[00]=1, W[11]=-1 -> k''=0,i=0:1; k''=1,i=1:-1
    
    # Indices: 0: k''=0,i=0; 1: k''=0,i=1; 2: k''=1,i=0; 3: k''=1,i=1
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # k''=0, i=0
        [0, 1, 0, 1, 0, 0, 0],  # k''=0, i=1
        [0, 0, 1, 0, 1, 0, 0],  # k''=1, i=0
        [1, 0, 1, 0, 0, 1, -1]  # k''=1, i=1
    ]
    
    # Fill global matrices
    for j_prime in range(n_j_groups):
        for k_prime in range(n_k_groups):
            base_col = (j_prime * n_k_groups + k_prime) * block_rank
            for r in range(block_rank):
                col = base_col + r
                for i in range(n):
                    for j_double in range(2):
                        val_u = U_loc[i * 2 + j_double][r]
                        if val_u != 0:
                            j = j_prime * 2 + j_double
                            u_idx = i * m + j
                            U[u_idx][col] += val_u
                            
                    for j_double in range(2):
                        for k_double in range(2):
                            val_v = V_loc[j_double * 2 + k_double][r]
                            if val_v != 0:
                                j = j_prime * 2 + j_double
                                k = k_prime * 2 + k_double
                                v_idx = j * p + k
                                V[v_idx][col] += val_v
                                
                    for k_double in range(2):
                        for i in range(n):
                            val_w = W_loc[k_double * 2 + i][r]
                            if val_w != 0:
                                k = k_prime * 2 + k_double
                                w_idx = k * n + i
                                W[w_idx][col] += val_w
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
