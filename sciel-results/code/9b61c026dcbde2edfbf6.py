"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    # Use block Strassen: split the 8 columns into 4 blocks of 2
    # Each block uses Strassen's rank-7 decomposition for 2x2 x 2x2
    num_blocks = 4
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen factors for local 2x2 x 2x2 multiplication
    # Local A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local W indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # U_local: 4 x 7, where row is A index, col is which M
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    
    # V_local: 4 x 7, where row is B index, col is which M
    V_local = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [1, 0, -1, -1, 0, 0, 1], # B01
        [1, 0, 1, -1, 0, 1, 0],  # B10
        [1, 0, 0, 0, 1, 0, 1],   # B11
    ]
    
    # W_local: 4 x 7, where row is C index, col is which M
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]
    
    for block in range(num_blocks):
        base_col = block * 7
        for m_idx in range(7):
            col = base_col + m_idx
            
            # Fill U: for each local A row (0-3), map to global U row
            for a_row in range(4):
                i = a_row // 2  # 0 or 1
                j = a_row % 2   # 0 or 1
                U[i * m + j][col] = U_local[a_row][m_idx]
            
            # Fill V: for each local B row (0-3), map to global V row
            # Local B row 0 -> B[0, 2*block], V row = 0*8 + 2*block
            # Local B row 1 -> B[0, 2*block+1], V row = 0*8 + 2*block+1
            # Local B row 2 -> B[1, 2*block], V row = 1*8 + 2*block
            # Local B row 3 -> B[1, 2*block+1], V row = 1*8 + 2*block+1
            for b_row in range(4):
                j = b_row // 2      # 0 or 1
                k_col = 2 * block + (b_row % 2)
                V[j * p + k_col][col] = V_local[b_row][m_idx]
            
            # Fill W: for each local C row (0-3), map to global W row
            # Local C row 0 -> C[0, 2*block], W row = 2*block*2 + 0
            # Local C row 1 -> C[0, 2*block+1], W row = (2*block+1)*2 + 0
            # Local C row 2 -> C[1, 2*block], W row = 2*block*2 + 1
            # Local C row 3 -> C[1, 2*block+1], W row = (2*block+1)*2 + 1
            for c_row in range(4):
                i = c_row // 2      # 0 or 1
                k_col = 2 * block + (c_row % 2)
                W[k_col * n + i][col] = W_local[c_row][m_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
