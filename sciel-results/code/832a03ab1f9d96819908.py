"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen U vectors (for 2x2 A)
    # A indices: 00->0, 01->1, 10->2, 11->3
    u_vectors = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # For each of the 4 blocks of B (and corresponding C block)
    for block_idx in range(4):
        # B is 2x8. Block k uses columns 2k and 2k+1.
        # B indices: j*8 + k.
        # b00 -> 2k, b01 -> 2k+1
        # b10 -> 8+2k, b11 -> 8+2k+1
        
        # C is 8x2. Block k uses rows 2k and 2k+1.
        # C indices: k*n + i = 2k + i.
        # c00 -> 4k, c01 -> 4k+2
        # c10 -> 4k+1, c11 -> 4k+3
        
        b_00_idx = 2 * block_idx
        b_01_idx = 2 * block_idx + 1
        b_10_idx = 8 + 2 * block_idx
        b_11_idx = 8 + 2 * block_idx + 1
        
        c_00_idx = 4 * block_idx
        c_01_idx = 4 * block_idx + 2
        c_10_idx = 4 * block_idx + 1
        c_11_idx = 4 * block_idx + 3
        
        # Define V and W vectors for this block
        # V vectors (for B part)
        v_vectors = [
            {b_00_idx: 1, b_11_idx: 1},     # M1: B00 + B11
            {b_00_idx: 1},                  # M2: B00
            {b_01_idx: 1, b_11_idx: -1},    # M3: B01 - B11
            {b_00_idx: 1, b_01_idx: -1},    # M4: B00 - B01
            {b_11_idx: 1},                  # M5: B11
            {b_00_idx: 1, b_01_idx: 1},     # M6: B00 + B01
            {b_10_idx: 1, b_11_idx: 1}      # M7: B10 + B11
        ]
        
        # W vectors (for C part)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        w_vectors = [
            {c_00_idx: 1, c_11_idx: 1},     # M1
            {c_10_idx: 1, c_11_idx: -1},    # M2
            {c_01_idx: 1, c_11_idx: 1},     # M3
            {c_00_idx: 1, c_11_idx: 1},     # M4
            {c_00_idx: -1, c_01_idx: 1},    # M5
            {c_11_idx: 1},                  # M6
            {c_00_idx: -1, c_10_idx: 1}     # M7
        ]
        
        # Fill U, V, W
        for t in range(7):
            col = block_idx * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][col] = u_vectors[t][i]
                
            # Fill V
            for idx, val in v_vectors[t].items():
                V[idx][col] = val
                
            # Fill W
            for idx, val in w_vectors[t].items():
                W[idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
