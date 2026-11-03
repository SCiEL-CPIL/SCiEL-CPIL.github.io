import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # We will build U, V, W as lists of lists, then convert to numpy arrays or just return lists.
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # Lists of lists are fine.
    
    # Part 1: 4x4x4 (cols 0-3 of B)
    # Part 2: 4x4x4 (cols 4-7 of B)
    # Part 3: 4x4x1 (col 8 of B)
    
    # Helper to get block indices
    # A blocks: A_00 (0,0), A_01 (0,1), A_10 (1,0), A_11 (1,1)
    # Global A index: i, j. Block i, j. Local u, v.
    # i = 2*bi + u, j = 2*bj + v.
    
    # Strassen block products
    # X_k and Y_k are 2x2 matrices of blocks.
    # We define them as dict of (bi, bj) -> coeff
    
    products = [
        # (X_coeffs, Y_coeffs)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),             # M1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),            # M2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),            # M3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),             # M4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),  # M5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})   # M6
    ]
    
    # Schoolbook for 2x2
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # For each block product k, we generate 4 terms.
    # Each term is (A_block, B_block, C_block, coeff_A, coeff_B, coeff_C)
    # Actually, the term is rank 1: U * V * W.
    # U has non-zero at A_block elements.
    # V has non-zero at B_block elements.
    # W has non-zero at C_block elements.
    
    # We need to map block indices to global indices.
    # A_block (bi, bj) -> global A rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # B_block (bi, bj) -> global B rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # C_block (ci, cj) -> global C rows 2*ci..2*ci+1, cols 2*cj..2*cj+1.
    
    # The C_block for the result of M_k is added to the final C.
    # Final C blocks:
    # C_00 = M0 - M3 + M4 - M6
    # C_01 = M2 + M4
    # C_10 = M1 + M3
    # C_11 = M0 - M2 + M1 - M5
    
    c_block_coeffs = [
        {(0,0): 1, (1,1): 1, (0,1): 1, (1,0): -1}, # M0? No, this is for M_k's contribution to C.
    ]
    
    # Let's define the contribution of each M_k to the C blocks.
    # M0 contributes to C_00 (1), C_11 (1)
    # M1 contributes to C_10 (1), C_11 (1)
    # M2 contributes to C_01 (1), C_11 (-1)
    # M3 contributes to C_00 (-1), C_10 (1)
    # M4 contributes to C_00 (1), C_01 (1)
    # M5 contributes to C_11 (-1)
    # M6 contributes to C_00 (-1)
    
    m_to_c = [
        {(0,0): 1, (1,1): 1}, # M0
        {(1,0): 1, (1,1): 1}, # M1
        {(0,1): 1, (1,1): -1}, # M2
        {(0,0): -1, (1,0): 1}, # M3
        {(0,0): 1, (0,1): 1}, # M4
        {(1,1): -1}, # M5
        {(0,0): -1} # M6
    ]
    
    U = []
    V = []
    W = []
    
    # We will accumulate terms.
    # Each term is a tuple of (U_vec, V_vec, W_vec)
    # But it's easier to just build the matrices directly.
    
    # Total rank estimate: 2 (4x4x4 parts) * 28 + 1 (4x4x1 part) * 16 = 72.
    # Let's pre-allocate or just append.
    
    # To handle the mapping, we create a function to add a term.
    # term: (a_block, b_block, c_block, coeff_a, coeff_b, coeff_c)
    # This adds a rank-1 term where:
    # U has coeff_a at the 4 elements of A_block.
    # V has coeff_b at the 4 elements of B_block.
    # W has coeff_c at the 4 elements of C_block.
    
    # Wait, this is not a single rank-1 term in the global tensor.
    # A rank-1 term in the global tensor has a single non-zero in U, V, W?
    # No, U, V, W are matrices.
    # A rank-1 term corresponds to one column in U, V, W.
    # In that column, U has non-zeros at the A indices, V at B indices, W at C indices.
    # If we use Schoolbook for the 2x2 block, we have 4 terms.
    # Term 1: A_00 * B_00. U has 1 at A_00 elements. V has 1 at B_00 elements. W has 1 at C_00 elements.
    # This is a valid rank-1 term.
    
    # So for each M_k, we generate 4 such terms.
    # For each term, we multiply the coefficients.
    
    for k in range(7):
        x_coeffs = products[k][0]
        y_coeffs = products[k][1]
        c_coeffs = m_to_c[k]
        
        # Schoolbook terms for 2x2
        # (a_block, b_block, c_block)
        sb_terms = [
            ((0,0), (0,0), (0,0)),
            ((0,1), (1,0), (0,0)),
            ((0,0), (0,1), (0,1)),
            ((0,1), (1,1), (0,1)),
            ((1,0), (0,0), (1,0)),
            ((1,1), (1,0), (1,0)),
            ((1,0), (0,1), (1,1)),
            ((1,1), (1,1), (1,1))
        ]
        
        for a_b, b_b, c_b in sb_terms:
            if a_b in x_coeffs and b_b in y_coeffs and c_b in c_coeffs:
                ca = x_coeffs[a_b]
                cb = y_coeffs[b_b]
                cc = c_coeffs[c_b]
                
                # Add term
                # U: non-zero at A_block a_b
                # V: non-zero at B_Block b_b
                # W: non-zero at C_Block c_b
                
                # We need to append to U, V, W.
                # Since we are building the matrices, we can just append rows later.
                # It's easier to collect the terms and then build the matrices.
                
                pass # Logic to be implemented in the final code

    # This approach is correct.
    # We will collect all terms (ca, cb, cc, a_b, b_b, c_b) for the 4x4x4 parts.
    # Then add the 4x4x1 part.
    # Then build U, V, W.

    return {"U": U, "V": V, "W": W}
