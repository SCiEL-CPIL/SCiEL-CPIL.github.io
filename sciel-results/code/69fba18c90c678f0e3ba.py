"""Program for rank-minimal tensor decomposition of 2x4 * 4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 4 matrix by a 4 by 7 matrix.
    
    Uses block decomposition with Strassen's algorithm for 2x2 blocks.
    Rank 50.
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
            
    # Helper to add Strassen block
    # M is 2x2 from A, N is 2x2 from B
    # u_base: index offset for M in U
    # v_base: index offset for N in V
    # w_base: index offset for result in W
    # col_offset: current column
    def add_strassen_block(u_base, v_base, w_base, col_offset):
        # M indices: u_base+0, u_base+1, u_base+4, u_base+5 (relative to A's 2x4 layout)
        # Actually, let's just pass the specific indices
        pass

    # We'll just hardcode the 50 terms as derived above for clarity and correctness.
    
    # Block 1: A1 B1
    # A1 cols 0,1. B1 rows 0,1.
    # B11 cols 0,1. B12 cols 2,3. B13 cols 4,5. B14 col 6.
    
    # Strassen 1.1 (cols 0,1)
    add_term(0, [0, 5], [1, 1], [0, 8], [1, 1], [0, 3], [1, 1])
    add_term(1, [4, 5], [1, 1], [0], [1], [1, 3], [1, -1])
    add_term(2, [0], [1], [1, 8], [1, -1], [2, 3], [1, 1])
    add_term(3, [5], [1], [7, 0], [1, -1], [0, 1], [-1, 1])
    add_term(4, [0, 1], [1, 1], [8], [1], [0, 3], [1, 1])
    add_term(5, [4, 0], [1, -1], [0, 1], [1, 1], [3], [1])
    add_term(6, [1, 5], [1, -1], [7, 8], [1, 1], [0, 2], [-1, 1])
    
    # Strassen 1.2 (cols 2,3)
    add_term(7, [0, 5], [1, 1], [2, 10], [1, 1], [4, 7], [1, 1])
    add_term(8, [4, 5], [1, 1], [2], [1], [5, 7], [1, -1])
    add_term(9, [0], [1], [3, 10], [1, -1], [6, 7], [1, 1])
    add_term(10, [5], [1], [9, 2], [1, -1], [4, 5], [-1, 1])
    add_term(11, [0, 1], [1, 1], [10], [1], [4, 7], [1, 1])
    add_term(12, [4, 0], [1, -1], [2, 3], [1, 1], [7], [1])
    add_term(13, [1, 5], [1, -1], [9, 10], [1, 1], [4, 6], [-1, 1])
    
    # Strassen 1.3 (cols 4,5)
    add_term(14, [0, 5], [1, 1], [4, 12], [1, 1], [8, 11], [1, 1])
    add_term(15, [4, 5], [1, 1], [4], [1], [9, 11], [1, -1])
    add_term(16, [0], [1], [5, 12], [1, -1], [10, 11], [1, 1])
    add_term(17, [5], [1], [11, 4], [1, -1], [8, 9], [-1, 1])
    add_term(18, [0, 1], [1, 1], [12], [1], [8, 11], [1, 1])
    add_term(19, [4, 0], [1, -1], [4, 5], [1, 1], [11], [1])
    add_term(20, [1, 5], [1, -1], [11, 12], [1, 1], [8, 10], [-1, 1])
    
    # Schoolbook 1.4 (col 6)
    add_term(21, [0], [1], [6], [1], [12], [1])
    add_term(22, [1], [1], [13], [1], [12], [1])
    add_term(23, [4], [1], [6], [1], [13], [1])
    add_term(24, [5], [1], [13], [1], [13], [1])
    
    # Block 2: A2 B2
    # A2 cols 2,3. B2 rows 2,3.
    # B21 cols 0,1. B22 cols 2,3. B23 cols 4,5. B24 col 6.
    
    # Strassen 2.1 (cols 0,1)
    add_term(25, [2, 7], [1, 1], [14, 22], [1, 1], [0, 3], [1, 1])
    add_term(26, [6, 7], [1, 1], [14], [1], [1, 3], [1, -1])
    add_term(27, [2], [1], [15, 22], [1, -1], [2, 3], [1, 1])
    add_term(28, [7], [1], [21, 14], [1, -1], [0, 1], [-1, 1])
    add_term(29, [2, 3], [1, 1], [22], [1], [0, 3], [1, 1])
    add_term(30, [6, 2], [1, -1], [14, 15], [1, 1], [3], [1])
    add_term(31, [3, 7], [1, -1], [21, 22], [1, 1], [0, 2], [-1, 1])
    
    # Strassen 2.2 (cols 2,3)
    add_term(32, [2, 7], [1, 1], [16, 24], [1, 1], [4, 7], [1, 1])
    add_term(33, [6, 7], [1, 1], [16], [1], [5, 7], [1, -1])
    add_term(34, [2], [1], [17, 24], [1, -1], [6, 7], [1, 1])
    add_term(35, [7], [1], [23, 16], [1, -1], [4, 5], [-1, 1])
    add_term(36, [2, 3], [1, 1], [24], [1], [4, 7], [1, 1])
    add_term(37, [6, 2], [1, -1], [16, 17], [1, 1], [7], [1])
    add_term(38, [3, 7], [1, -1], [23, 24], [1, 1], [4, 6], [-1, 1])
    
    # Strassen 2.3 (cols 4,5)
    add_term(39, [2, 7], [1, 1], [18, 26], [1, 1], [8, 11], [1, 1])
    add_term(40, [6, 7], [1, 1], [18], [1], [9, 11], [1, -1])
    add_term(41, [2], [1], [19, 26], [1, -1], [10, 11], [1, 1])
    add_term(42, [7], [1], [25, 18], [1, -1], [8, 9], [-1, 1])
    add_term(43, [2, 3], [1, 1], [26], [1], [8, 11], [1, 1])
    add_term(44, [6, 2], [1, -1], [18, 19], [1, 1], [11], [1])
    add_term(45, [3, 7], [1, -1], [25, 26], [1, 1], [8, 10], [-1, 1])
    
    # Schoolbook 2.4 (col 6)
    add_term(46, [2], [1], [20], [1], [12], [1])
    add_term(47, [3], [1], [27], [1], [12], [1])
    add_term(48, [6], [1], [20], [1], [13], [1])
    add_term(49, [7], [1], [27], [1], [13], [1])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
