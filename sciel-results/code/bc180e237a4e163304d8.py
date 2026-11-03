"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor using a block-Strassen approach.
    
    Strategy:
    - Split A (2x3) into A_left (2x2) and A_right (2x1).
    - Split B (3x8) into B_top (2x8) and B_bottom (1x8).
    - C = A_left * B_top + A_right * B_bottom.
    
    For A_left * B_top (2x2 x 2x8):
    - Split B_top into four 2x2 blocks: B_0, B_1, B_2, B_3.
    - Apply Strassen (rank 7) to each 2x2x2 multiplication.
    - Total rank for this part: 4 * 7 = 28.
    
    For A_right * B_bottom (2x1 x 1x8):
    - This is an outer product.
    - Requires 2 * 8 = 16 multiplications.
    
    Total rank: 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_indices: list of (index, coeff) for U
    # v_indices: list of (index, coeff) for V
    # w_indices: list of (index, coeff) for W
    def add_term(col, u_indices, v_indices, w_indices):
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff

    # Strassen parameters for 2x2 block
    # A indices: 00->0, 01->1, 10->3, 11->4
    # B indices (relative to block start col c, rows 0,1):
    #   00 -> 0*8 + c
    #   01 -> 0*8 + c+1
    #   10 -> 1*8 + c
    #   11 -> 1*8 + c+1
    # C indices (relative to block start col c, rows 0,1):
    #   00 -> (c)*2 + 0
    #   01 -> (c+1)*2 + 0
    #   10 -> (c)*2 + 1
    #   11 -> (c+1)*2 + 1
    
    col = 0
    for b in range(4):
        c = 4 * b
        
        # B indices
        b11_idx = 0 * 8 + c
        b12_idx = 0 * 8 + (c + 1)
        b21_idx = 1 * 8 + c
        b22_idx = 1 * 8 + (c + 1)
        
        # C indices
        c11_idx = c * 2 + 0
        c12_idx = (c + 1) * 2 + 0
        c21_idx = c * 2 + 1
        c22_idx = (c + 1) * 2 + 1
        
        # M1 = (A11 + A22) * (B11 + B22)
        add_term(col, 
                 [(0, 1), (4, 1)], 
                 [(b11_idx, 1), (b22_idx, 1)], 
                 [(c11_idx, 1), (c22_idx, 1)])
        col += 1
        
        # M2 = (A21 + A22) * B11
        add_term(col, 
                 [(3, 1), (4, 1)], 
                 [(b11_idx, 1)], 
                 [(c21_idx, 1), (c22_idx, -1)])
        col += 1
        
        # M3 = A11 * (B12 - B22)
        add_term(col, 
                 [(0, 1)], 
                 [(b12_idx, 1), (b22_idx, -1)], 
                 [(c12_idx, 1), (c22_idx, 1)])
        col += 1
        
        # M4 = A22 * (B21 - B11)
        add_term(col, 
                 [(4, 1)], 
                 [(b21_idx, 1), (b11_idx, -1)], 
                 [(c11_idx, 1), (c21_idx, 1)])
        col += 1
        
        # M5 = (A11 + A12) * B22
        add_term(col, 
                 [(0, 1), (1, 1)], 
                 [(b22_idx, 1)], 
                 [(c11_idx, -1), (c12_idx, 1)])
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        add_term(col, 
                 [(3, 1), (0, -1)], 
                 [(b11_idx, 1), (b12_idx, 1)], 
                 [(c22_idx, 1)])
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        add_term(col, 
                 [(1, 1), (4, -1)], 
                 [(b21_idx, 1), (b22_idx, 1)], 
                 [(c11_idx, 1)])
        col += 1
        
    # A_right * B_bottom
    # A_right: col 2 of A. Indices: 0*3+2=2, 1*3+2=5
    # B_bottom: row 2 of B. Indices: 2*8+k = 16+k
    # C: rows 0,1. Indices: k*2+0, k*2+1
    
    for k in range(8):
        # C_0k += A_02 * B_2k
        add_term(col, 
                 [(2, 1)], 
                 [(16 + k, 1)], 
                 [(k * 2, 1)])
        col += 1
        
        # C_1k += A_12 * B_2k
        add_term(col, 
                 [(5, 1)], 
                 [(16 + k, 1)], 
                 [(k * 2 + 1, 1)])
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
