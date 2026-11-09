"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition 5 = 2 + 2 + 1
    # The 4x4 core is handled by Strassen (rank 49)
    # The remaining terms are handled by schoolbook
    
    # Strassen decomposition for 2x2 matrices
    # Indices: 0,1 for first block, 2,3 for second block, 4 for third block
    # U, V, W matrices will be built as lists of lists
    
    rank = 94
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Strassen for 4x4 core (blocks 0,1 and 2,3)
    # The 7 Strassen products for 2x2 blocks
    # Each product is of the form (linear combo of A) * (linear combo of B)
    # We need to expand these into scalar multiplications
    
    # Strassen variables for the 4x4 block multiplication
    # A11, A12, A21, A22 are 2x2 blocks
    # B11, B12, B21, B22 are 2x2 blocks
    
    # M1 = (A11 + A22)(B11 + B22)
    # This involves 4x4 = 16 scalar multiplications? No, it's a 2x2 matrix product.
    # The rank of a 2x2 matrix product is 7 (Strassen).
    # So we need to add 7 terms for M1.
    
    # We will use the Strassen decomposition for each 2x2 product.
    # This is a nested structure.
    
    # Let's define the Strassen terms for a 2x2 product (i,j) and (k,l)
    # where i,j are row indices of A block, k,l are row indices of B block?
    # No, A is n x m, B is m x p.
    # For the 4x4 core:
    # A rows 0-3, B cols 0-3.
    # A11: rows 0-1, cols 0-1. B11: rows 0-1, cols 0-1.
    
    def add_strassen_2x2(a_row_start, a_col_start, b_row_start, b_col_start):
        nonlocal col
        # a_row_start, a_col_start: top-left of A block (2x2)
        # b_row_start, b_col_start: top-left of B block (2x2)
        
        # A11, A12, A21, A22 relative to block
        # B11, B12, B21, B22 relative to block
        
        # M1 = (A11 + A22)(B11 + B22)
        # This is a 2x2 matrix product. We use Strassen for it.
        # The Strassen algorithm for 2x2 uses 7 multiplications.
        # We add 7 terms to the global rank.
        
        # We need to map the local Strassen variables to global indices.
        # This is complex. Let's just use schoolbook for the 2x2 products?
        # If we use schoolbook for 2x2, rank is 4.
        # 7 * 4 = 28 for the 4x4 core.
        # Plus 45 for the rest. Total 73.
        # This is better than 94.
        
        # Let's use schoolbook for the 2x2 blocks.
        # 2x2 schoolbook: 4 terms.
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a_idx = (a_row_start + i) * m + (a_col_start + j)
                    b_idx = (b_row_start + i) * p + (b_col_start + j) # Wait, B is m x p.
                    # B row index is j (from A col), B col index is k.
                    # B row is j (0-1), B col is k (0-1).
                    # Global B row: b_row_start + j. Global B col: b_col_start + k.
                    b_idx = (b_row_start + j) * p + (b_col_start + k)
                    c_idx = (b_col_start + k) * n + (a_row_start + i)
                    
                    add_term(a_idx, b_idx, c_idx)

    # Core 4x4: 4 blocks of 2x2
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    
    # We need to multiply the 4x4 blocks.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Each term is a 2x2 x 2x2 product.
    # We use schoolbook for these (rank 4 each).
    # 8 terms * 4 = 32.
    
    # C11
    add_strassen_2x2(0, 0, 0, 0) # A11 B11
    add_strassen_2x2(0, 2, 2, 0) # A12 B21
    
    # C12
    add_strassen_2x2(0, 0, 0, 2) # A11 B12
    add_strassen_2x2(0, 2, 2, 2) # A12 B22
    
    # C21
    add_strassen_2x2(2, 0, 0, 0) # A21 B11
    add_strassen_2x2(2, 2, 2, 0) # A22 B21
    
    # C22
    add_strassen_2x2(2, 0, 0, 2) # A21 B12
    add_strassen_2x2(2, 2, 2, 2) # A22 B22
    
    # Now the cross terms with the 5th row/col (index 4)
    # C15 = A11 B15 + A12 B25 + A13 B35 (A13 is 2x1, B15 is 2x1? No, B15 is 2x1? No, B is 5x5. B15 is rows 0-1, col 4.)
    # A11 (2x2) * B15 (2x1) -> 2x1. Rank 4.
    # A12 (2x2) * B25 (2x1) -> 2x1. Rank 4.
    # A13 (2x1) * B35 (1x1) -> 2x1. Rank 2.
    
    # C15
    add_strassen_2x2(0, 0, 0, 4) # A11 B15 (B col 4)
    add_strassen_2x2(0, 2, 2, 4) # A12 B25
    
    # A13 B35: A rows 0-1, col 4. B row 4, col 4.
    for i in range(2):
        a_idx = i * m + 4
        b_idx = 4 * p + 4
        c_idx = 4 * n + i
        add_term(a_idx, b_idx, c_idx)
        
    # C25
    add_strassen_2x2(2, 0, 0, 4) # A21 B15
    add_strassen_2x2(2, 2, 2, 4) # A22 B25
    
    for i in range(2, 4):
        a_idx = i * m + 4
        b_idx = 4 * p + 4
        c_idx = 4 * n + i
        add_term(a_idx, b_idx, c_idx)

    # C51
    # A51 (1x2) * B11 (2x2) -> 1x2. Rank 4.
    # A52 (1x2) * B21 (2x2) -> 1x2. Rank 4.
    # A53 (1x1) * B31 (1x2) -> 1x2. Rank 2.
    
    add_strassen_2x2(4, 0, 0, 0) # A51 B11 (A row 4)
    add_strassen_2x2(4, 2, 2, 0) # A52 B21
    
    for j in range(2):
        a_idx = 4 * m + j
        b_idx = 4 * p + j
        c_idx = j * n + 4
        add_term(a_idx, b_idx, c_idx)

    # C52
    add_strassen_2x2(4, 0, 0, 2) # A51 B12
    add_strassen_2x2(4, 2, 2, 2) # A52 B22
    
    for j in range(2, 4):
        a_idx = 4 * m + j
        b_idx = 4 * p + j
        c_idx = j * n + 4
        add_term(a_idx, b_idx, c_idx)

    # C55
    # A51 (1x2) * B15 (2x1) -> 1x1. Rank 2.
    # A52 (1x2) * B25 (2x1) -> 1x1. Rank 2.
    # A53 (1x1) * B35 (1x1) -> 1x1. Rank 1.
    
    for k in range(2):
        a_idx = 4 * m + k
        b_idx = k * p + 4
        c_idx = 4 * n + 4
        add_term(a_idx, b_idx, c_idx)
        
    for k in range(2, 4):
        a_idx = 4 * m + k
        b_idx = k * p + 4
        c_idx = 4 * n + 4
        add_term(a_idx, b_idx, c_idx)
        
    a_idx = 4 * m + 4
    b_idx = 4 * p + 4
    c_idx = 4 * n + 4
    add_term(a_idx, b_idx, c_idx)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
