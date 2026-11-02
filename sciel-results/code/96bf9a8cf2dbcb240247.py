import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a block decomposition based on Strassen's algorithm for M_{2,2,2}.
    Rank = 28.
    """
    n, m, p = DIMENSIONS
    
    # Define the 7 Strassen terms for M_{2,2,2}
    # U_vecs: (7, 4)
    # V_vecs: (7, 4)
    # W_vecs: (7, 4)
    # Indexing for 2x2 matrices: [00, 01, 10, 11]
    
    u_strassen = np.array([
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ], dtype=float)
    
    v_strassen = np.array([
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ], dtype=float)
    
    w_strassen = np.array([
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ], dtype=float)
    
    # Total rank
    rank = 28
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float) # (8, 28)
    V = np.zeros((m * p, rank), dtype=float) # (16, 28)
    W = np.zeros((p * n, rank), dtype=float) # (8, 28)
    
    # Helper to map 2x2 indices to global indices
    # A is 2x4. Global U index: i * 4 + j.
    # B is 4x4. Global V index: j * 4 + k.
    # C is 2x4. Global W index: k * 2 + i.
    
    # We process two blocks: Left (cols 0,1 of B) and Right (cols 2,3 of B)
    
    for block in range(2):
        # block 0: B_left (cols 0,1)
        # block 1: B_right (cols 2,3)
        
        # Offset for B columns
        b_col_offset = block * 2
        
        # Offset for C columns (output)
        c_col_offset = block * 2
        
        # We use 7 Strassen terms for A1 * B_block_top + A2 * B_block_bottom
        # This corresponds to M_{2,4,2} decomposition.
        # It consists of 2 sub-multiplications of M_{2,2,2}.
        # Sub-mult 1: A1 * B_top
        # Sub-mult 2: A2 * B_bottom
        
        for sub_block in range(2):
            # sub_block 0: A1 (cols 0,1 of A) * B_top (rows 0,1 of B_block)
            # sub_block 1: A2 (cols 2,3 of A) * B_bottom (rows 2,3 of B_block)
            
            # A column offset
            a_col_offset = sub_block * 2
            
            # B row offset (within the 4x4 B)
            # B_block has rows 0,1 for top, 2,3 for bottom.
            # So B row offset is sub_block * 2.
            b_row_offset = sub_block * 2
            
            # Start column index in U, V, W
            col_start = block * 14 + sub_block * 7
            
            for s in range(7):
                col_idx = col_start + s
                
                # U: Map 2x2 A entries to global U
                # A entries are at (i, j) where i in {0,1}, j in {0,1} for A1 or {2,3} for A2.
                # Global U index: i * 4 + (j + a_col_offset)
                for i in range(2):
                    for j in range(2):
                        local_idx = i * 2 + j
                        global_u_row = i * 4 + (j + a_col_offset)
                        U[global_u_row, col_idx] = u_strassen[s, local_idx]
                        
                # V: Map 2x2 B entries to global V
                # B entries are at (j, k) where j in {0,1} for top or {2,3} for bottom, k in {0,1} for left or {2,3} for right.
                # Global V index: (j + b_row_offset) * 4 + (k + b_col_offset)
                for j in range(2):
                    for k in range(2):
                        local_idx = j * 2 + k
                        global_v_row = (j + b_row_offset) * 4 + (k + b_col_offset)
                        V[global_v_row, col_idx] = v_strassen[s, local_idx]
                        
                # W: Map 2x2 C entries to global W
                # C entries are at (i, k) where i in {0,1}, k in {0,1} for left or {2,3} for right.
                # Global W index: (k + c_col_offset) * 2 + i
                for i in range(2):
                    for k in range(2):
                        local_idx = k * 2 + i
                        global_w_row = (k + c_col_offset) * 2 + i
                        W[global_w_row, col_idx] = w_strassen[s, local_idx]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
