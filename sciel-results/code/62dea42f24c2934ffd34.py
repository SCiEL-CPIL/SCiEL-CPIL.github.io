"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 independent Strassen (rank 7) blocks
    rank = 28
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition vectors for 2x2x2
    # u: length 4 (i*m+j), v: length 4 (j*p+k), w: length 4 (k*n+i)
    # Indices for 2x2x2:
    # u: 00, 01, 10, 11
    # v: 00, 01, 10, 11
    # w: 00, 01, 10, 11 (k*n+i)
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, -1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    strassen_w = [
        [1, 1, 1, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]
    
    # Define the 4 blocks
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2] (k in 0,1)
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] (k in 0,1)
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4] (k in 2,3)
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] (k in 2,3)
    
    blocks = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0), # A0, B00, CL
        (2, 2, 0, 0), # A1, B10, CL
        (0, 0, 2, 2), # A0, B01, CR
        (2, 2, 2, 2), # A1, B11, CR
    ]
    
    for b_idx, (a_off, b_row_off, b_col_off, c_col_off) in enumerate(blocks):
        base_rank = b_idx * 7
        for r in range(7):
            col = base_rank + r
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_val = strassen_u[r][i*2 + j]
                    if u_val != 0:
                        U[i*m + (j + a_off)][col] += u_val
                        
            # Fill V
            for j in range(2):
                for k in range(2):
                    v_val = strassen_v[r][j*2 + k]
                    if v_val != 0:
                        V[(j + b_row_off)*p + (k + b_col_off)][col] += v_val
                        
            # Fill W
            for k in range(2):
                for i in range(2):
                    w_val = strassen_w[r][k*2 + i]
                    if w_val != 0:
                        W[(k + c_col_off)*n + i][col] += w_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
