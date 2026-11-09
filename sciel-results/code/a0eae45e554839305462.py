import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Partition indices into blocks of sizes 2, 2, 1
    # Indices: 0,1 | 2,3 | 4
    partitions = [[0, 1], [2, 3], [4]]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add schoolbook decomposition for n x m x p
    def add_schoolbook(n, m, p, U_offset, V_offset, W_offset):
        rank = n * m * p
        U_block = [[0] * rank for _ in range(n * m)]
        V_block = [[0] * rank for _ in range(m * p)]
        W_block = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U_block[i * m + j][col] = 1
                    V_block[j * p + k][col] = 1
                    W_block[k * n + i][col] = 1
                    col += 1
                    
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)

    # Helper to add Strassen decomposition for 2 x 2 x 2
    def add_strassen(U_offset, V_offset, W_offset):
        # U, V, W shapes (4, 7)
        rank = 7
        U_block = [[0] * rank for _ in range(4)]
        V_block = [[0] * rank for _ in range(4)]
        W_block = [[0] * rank for _ in range(4)]
        
        # m1 = (a+d)(e+h)
        U_block[0][0] = 1; U_block[3][0] = 1
        V_block[0][0] = 1; V_block[3][0] = 1
        W_block[0][0] = 1; W_block[3][0] = 1
        
        # m2 = (c+d)e
        U_block[2][1] = 1; U_block[3][1] = 1
        V_block[0][1] = 1
        W_block[2][1] = 1; W_block[3][1] = 1
        
        # m3 = a(f-h)
        U_block[0][2] = 1
        V_block[1][2] = 1; V_block[3][2] = -1
        W_block[1][2] = 1; W_block[3][2] = 1
        
        # m4 = d(g-e)
        U_block[3][3] = 1
        V_block[2][3] = 1; V_block[0][3] = -1
        W_block[0][3] = 1; W_block[2][3] = 1
        
        # m5 = (a+b)h
        U_block[0][4] = 1; U_block[1][4] = 1
        V_block[3][4] = 1
        W_block[0][4] = -1; W_block[1][4] = 1
        
        # m6 = (c-a)(e+f)
        U_block[2][5] = 1; U_block[0][5] = -1
        V_block[0][5] = 1; V_block[1][5] = 1
        W_block[3][5] = 1
        
        # m7 = (b-d)(g+h)
        U_block[1][6] = 1; U_block[3][6] = -1
        V_block[2][6] = 1; V_block[3][6] = 1
        W_block[0][6] = 1; W_block[3][6] = -1
        
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)

    # Iterate over block indices i, j, k
    for i_idx, I in enumerate(partitions):
        for j_idx, J in enumerate(partitions):
            for k_idx, K in enumerate(partitions):
                n_i = len(I)
                n_j = len(J)
                n_k = len(K)
                
                # Check if we can use Strassen (2x2x2)
                if n_i == 2 and n_j == 2 and n_k == 2:
                    # We need to map the local 2x2x2 decomposition to the global indices
                    # This requires shifting the indices in U, V, W
                    # However, the add_strassen function generates a block for a 2x2x2 tensor.
                    # We need to place it in the correct position in the global U, V, W.
                    
                    # For now, let's just append the block. 
                    # Wait, the global U has shape (5*5, Rank). 
                    # The block U has shape (2*2, 7).
                    # We need to embed this block into the global U at the correct row offsets.
                    
                    # Let's modify the approach: 
                    # Instead of appending lists of blocks and concatenating later, 
                    # we should build the full U, V, W matrices directly or use a helper that handles offsets.
                    
                    pass # Placeholder, we will rewrite the loop to handle offsets properly

    # Rewrite: Build U, V, W directly
    total_rank = 0
    # First pass: calculate total rank
    for i_idx, I in enumerate(partitions):
        for j_idx, J in enumerate(partitions):
            for k_idx, K in enumerate(partitions):
                n_i = len(I)
                n_j = len(J)
                n_k = len(K)
                if n_i == 2 and n_j == 2 and n_k == 2:
                    total_rank += 7
                else:
                    total_rank += n_i * n_j * n_k

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    for i_idx, I in enumerate(partitions):
        for j_idx, J in enumerate(partitions):
            for k_idx, K in enumerate(partitions):
                n_i = len(I)
                n_j = len(J)
                n_k = len(K)
                
                if n_i == 2 and n_j == 2 and n_k == 2:
                    # Strassen
                    # Local indices: 0,1 in I, J, K
                    # Global indices: I[0], I[1], etc.
                    
                    # U offset: I[0]*m + J[0], etc.
                    # V offset: J[0]*p + K[0], etc.
                    # W offset: K[0]*n + I[0], etc.
                    
                    # We generate the 4x7 blocks and place them
                    U_block, V_block, W_block = _get_strassen_blocks()
                    
                    # Place U_block
                    for local_u_row in range(4):
                        global_u_row = I[local_u_row // 2] * m + J[local_u_row % 2]
                        for local_col in range(7):
                            U[global_u_row][current_rank + local_col] = U_block[local_u_row][local_col]
                            
                    # Place V_block
                    for local_v_row in range(4):
                        global_v_row = J[local_v_row // 2] * p + K[local_v_row % 2]
                        for local_col in range(7):
                            V[global_v_row][current_rank + local_col] = V_block[local_v_row][local_col]
                            
                    # Place W_block
                    for local_w_row in range(4):
                        global_w_row = K[local_w_row // 2] * n + I[local_w_row % 2]
                        for local_col in range(7):
                            W[global_w_row][current_rank + local_col] = W_block[local_w_row][local_col]
                            
                    current_rank += 7

                else:
                    # Schoolbook
                    rank_block = n_i * n_j * n_k
                    col = 0
                    for li in range(n_i):
                        for lj in range(n_j):
                            for lk in range(n_k):
                                global_u_row = I[li] * m + J[lj]
                                global_v_row = J[lj] * p + K[lk]
                                global_w_row = K[lk] * n + I[li]
                                
                                U[global_u_row][current_rank + col] = 1
                                V[global_v_row][current_rank + col] = 1
                                W[global_w_row][current_rank + col] = 1
                                col += 1
                    current_rank += rank_block

    return {"U": U, "V": V, "W": W}

def _get_strassen_blocks():
    rank = 7
    U_block = [[0] * rank for _ in range(4)]
    V_block = [[0] * rank for _ in range(4)]
    W_block = [[0] * rank for _ in range(4)]
    
    # m1
    U_block[0][0] = 1; U_block[3][0] = 1
    V_block[0][0] = 1; V_block[3][0] = 1
    W_block[0][0] = 1; W_block[3][0] = 1
    
    # m2
    U_block[2][1] = 1; U_block[3][1] = 1
    V_block[0][1] = 1
    W_block[2][1] = 1; W_block[3][1] = 1
    
    # m3
    U_block[0][2] = 1
    V_block[1][2] = 1; V_block[3][2] = -1
    W_block[1][2] = 1; W_block[3][2] = 1
    
    # m4
    U_block[3][3] = 1
    V_block[2][3] = 1; V_block[0][3] = -1
    W_block[0][3] = 1; W_block[2][3] = 1
    
    # m5
    U_block[0][4] = 1; U_block[1][4] = 1
    V_block[3][4] = 1
    W_block[0][4] = -1; W_block[1][4] = 1
    
    # m6
    U_block[2][5] = 1; U_block[0][5] = -1
    V_block[0][5] = 1; V_block[1][5] = 1
    W_block[3][5] = 1
    
    # m7
    U_block[1][6] = 1; U_block[3][6] = -1
    V_block[2][6] = 1; V_block[3][6] = 1
    W_block[0][6] = 1; W_block[3][6] = -1
    
    return U_block, V_block, W_block

# EVOLVE-BLOCK-END
