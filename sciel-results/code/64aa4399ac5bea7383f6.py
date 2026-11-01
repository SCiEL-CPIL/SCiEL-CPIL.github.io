"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using 5 blocks of Strassen (rank 7 each)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # U: 4 elements (a00, a01, a10, a11)
    # V: 4 elements (b00, b01, b10, b11)
    # W: 4 elements (c00, c01, c10, c11)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [-1, 1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, -1],# M1
        [0, 0, 1, 1], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 0, 0], # M4
        [0, 0, 1, 0], # M5
        [1, 0, 0, 0], # M6
        [0, 0, 0, -1] # M7
    ]
    
    # Combine for C = M1 - M4 + M5 + M6, etc.
    # C00 = 1*M1 - 1*M4 + 1*M5 + 1*M6
    # C01 = 1*M3 + 1*M4
    # C10 = 1*M2 + 1*M5
    # C11 = 1*M1 - 1*M3 + 1*M2 - 1*M7
    
    # We need to sum the W vectors accordingly for each term in the final decomposition?
    # No, the decomposition is a sum of rank-1 tensors.
    # Each Strassen term r is a rank-1 tensor U_r \otimes V_r \otimes W_r.
    # The final tensor is sum of 7 such tensors.
    # So we just place U_r, V_r, W_r into the columns.
    
    num_blocks = 5
    for k in range(num_blocks):
        for r in range(7):
            col = k * 7 + r
            
            # U is same for all blocks
            for i in range(4):
                U[i][col] = S_U[r][i]
                
            # V indices for block k
            # b00 -> 2k, b01 -> 2k+1, b10 -> 10+2k, b11 -> 10+2k+1
            v_indices = [2*k, 2*k+1, 10+2*k, 10+2*k+1]
            for i in range(4):
                V[v_indices[i]][col] = S_V[r][i]
                
            # W indices for block k
            # c00 -> 4k, c01 -> 4k+2, c10 -> 4k+1, c11 -> 4k+3
            # Note: W index is k*n+i. 
            # Block k covers C columns 2k, 2k+1.
            # C_{0, 2k} -> (2k)*2 + 0 = 4k
            # C_{0, 2k+1} -> (2k+1)*2 + 0 = 4k+2
            # C_{1, 2k} -> (2k)*2 + 1 = 4k+1
            # C_{1, 2k+1} -> (2k+1)*2 + 1 = 4k+3
            w_indices = [4*k, 4*k+2, 4*k+1, 4*k+3]
            for i in range(4):
                W[w_indices[i]][col] = S_W[r][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
