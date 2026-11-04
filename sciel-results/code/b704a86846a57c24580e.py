"""Program for matrix multiplication tensor rank minimization (3x4 by 4x6)."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: 2x2x2 Strassen blocks + residual trivial part
    # Partitions: i in {0,1} for Strassen, i=2 for residual
    # j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    # Strassen 2x2x2 factors (local indices)
    # U_local: 4 rows (i_l*m_l + j_l for i_l,j_l in {0,1}), 7 cols
    # V_local: 4 rows (j_l*p_l + k_l), 7 cols
    # W_local: 4 rows (k_l*n_l + i_l), 7 cols
    
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],   # a: i_l=0, j_l=0
        [0, 0, 0, 0, 1,  0, 1],   # b: i_l=0, j_l=1
        [0, 1, 0, 0, 0,  1, 0],   # c: i_l=1, j_l=0
        [1, 1, 0, 1, 0,  0,-1],   # d: i_l=1, j_l=1
    ]
    V_local = [
        [1, 1, 0, 0, 0, 1, 0],   # e: j_l=0, k_l=0
        [1, 0, 1, 0, 0, 1, 0],   # f: j_l=0, k_l=1
        [1, 0, 0, 1, 0, 0, 1],   # g: j_l=1, k_l=0
        [1, 0,-1, 0, 1, 0, 1],   # h: j_l=1, k_l=1
    ]
    W_local = [
        [1, 0, 0,-1, 1, 0, 1],   # C[0,0]: k_l=0, i_l=0
        [0, 0, 1, 1, 0, 0, 0],   # C[0,1]: k_l=1, i_l=0
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]: k_l=0, i_l=1
        [1,-1, 1, 0, 0, 1, 0],   # C[1,1]: k_l=1, i_l=1
    ]
    
    # Define block partitions
    # For i: use 2 rows for Strassen (i_base=0, so i_l in {0,1} -> i in {0,1}), residual i=2
    # For j: 4 cols, partition into 2 blocks of 2: j_base in {0, 2}
    # For k: 6 cols, partition into 3 blocks of 2: k_base in {0, 2, 4}
    
    total_rank = 0
    # Strassen blocks: 2 (j_blocks) * 3 (k_blocks) = 6 blocks, each rank 7
    # Residual: i=2, all j (4), all k (6) = 24 terms
    total_rank = 6 * 7 + 24
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Add Strassen blocks
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            for s in range(7):
                # U: rows for i_l in {0,1}, j_l in {0,1}
                for i_l in range(2):
                    for j_l in range(2):
                        i = 0 + i_l  # i_base = 0 for Strassen part
                        j = j_base + j_l
                        row_idx = i * m + j
                        U[row_idx][col] += U_local[i_l * 2 + j_l][s]
                
                # V: rows for j_l in {0,1}, k_l in {0,1}
                for j_l in range(2):
                    for k_l in range(2):
                        j = j_base + j_l
                        k = k_base + k_l
                        row_idx = j * p + k
                        V[row_idx][col] += V_local[j_l * 2 + k_l][s]
                
                # W: rows for k_l in {0,1}, i_l in {0,1}
                for k_l in range(2):
                    for i_l in range(2):
                        k = k_base + k_l
                        i = 0 + i_l
                        row_idx = k * n + i
                        W[row_idx][col] += W_local[k_l * 2 + i_l][s]
                
                col += 1
    
    # Add residual part: i=2, all j, all k (trivial decomposition)
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
