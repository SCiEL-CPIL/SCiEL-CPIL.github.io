"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition with Strassen for the 2x2 part.
    # A is 3x3, B is 3x8.
    # Split A into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1).
    # Split B into B11 (2x4), B12 (2x4), B21 (1x4), B22 (1x4).
    # B11 is split into B11a (2x2) and B11b (2x2).
    # B12 is split into B12a (2x2) and B12b (2x2).
    
    U = []
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        """
        u_indices: list of (i, j) for A
        v_indices: list of (j, k) for B
        w_indices: list of (k, i) for C
        coeff: multiplier for the term
        """
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for i, j in u_indices:
            col_u[i * m + j] += coeff
        for j, k in v_indices:
            col_v[j * p + k] += coeff
        for k, i in w_indices:
            col_w[k * n + i] += coeff
            
        U.append(col_u)
        V.append(col_v)
        W.append(col_w)

    # Strassen terms for 2x2 * 2x2
    # A11 is rows 0,1 cols 0,1
    # B block is rows 0,1 cols c0, c0+1
    
    def strassen_2x2(c0):
        # B block is B[0:2, c0:c0+2]
        # M1 = (A00 + A11)(B00 + B11)
        add_term([(0,0), (1,1)], [(0,c0), (1,c0+1)], [(c0,0), (c0+1,1)])
        # M2 = (A10 + A11)B00
        add_term([(1,0), (1,1)], [(0,c0)], [(c0,1)])
        # M3 = A00(B01 - B11)
        add_term([(0,0)], [(0,c0+1), (1,c0+1)], [(c0+1,0)], coeff=1)
        # Wait, M3 = A00 * (B01 - B11).
        # The term is A00*B01 - A00*B11.
        # We can add two terms or one term with coeff -1?
        # The decomposition is sum of products.
        # If we use one column, V must be linear combination.
        # V has 1 at (0, c0+1) and -1 at (1, c0+1).
        # W has 1 at (c0+1, 0).
        # This works.
        add_term([(0,0)], [(0,c0+1), (1,c0+1)], [(c0+1,0)], coeff=1)
        # Correction: The previous line was a duplicate or wrong.
        # Let's restart M3.
        # I already added M1, M2.
        # M3 = A00(B01 - B11)
        # U: (0,0)
        # V: (0,c0+1) coeff 1, (1,c0+1) coeff -1
        # W: (c0+1,0) coeff 1
        # But add_term only takes one coeff for all.
        # I need to support different coeffs in V?
        # No, the rank-1 term is U * V * W.
        # If V has different coeffs, it's still one rank-1 term.
        # My add_term function adds `coeff` to all.
        # I need to modify add_term or call it multiple times.
        # It's easier to just add multiple rank-1 terms if needed, but that increases rank.
        # Strassen relies on linear combinations in U, V, W.
        # So one column can have different values in V.
        # My add_term function is too rigid.
        pass

    # Let's implement a more flexible add_term
    def add_term_flex(u_dict, v_dict, w_dict):
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for (i, j), val in u_dict.items():
            col_u[i * m + j] += val
        for (j, k), val in v_dict.items():
            col_v[j * p + k] += val
        for (k, i), val in w_dict.items():
            col_w[k * n + i] += val
            
        U.append(col_u)
        V.append(col_v)
        W.append(col_w)

    # Re-implement Strassen
    def strassen_2x2(c0):
        # M1 = (A00 + A11)(B00 + B11)
        add_term_flex(
            {(0,0): 1, (1,1): 1},
            {(0,c0): 1, (1,c0+1): 1},
            {(c0,0): 1, (c0+1,1): 1}
        )
        # M2 = (A10 + A11)B00
        add_term_flex(
            {(1,0): 1, (1,1): 1},
            {(0,c0): 1},
            {(c0,1): 1}
        )
        # M3 = A00(B01 - B11)
        add_term_flex(
            {(0,0): 1},
            {(0,c0+1): 1, (1,c0+1): -1},
            {(c0+1,0): 1}
        )
        # M4 = A11(B10 - B00)
        add_term_flex(
            {(1,1): 1},
            {(1,c0): 1, (0,c0): -1},
            {(c0,1): 1}
        )
        # M5 = (A00 + A01)B11
        add_term_flex(
            {(0,0): 1, (0,1): 1},
            {(1,c0+1): 1},
            {(c0+1,0): 1}
        )
        # M6 = (A10 - A00)(B00 + B01)
        add_term_flex(
            {(1,0): 1, (0,0): -1},
            {(0,c0): 1, (0,c0+1): 1},
            {(c0+1,1): 1}
        )
        # M7 = (A01 - A11)(B10 + B11)
        add_term_flex(
            {(0,1): 1, (1,1): -1},
            {(1,c0): 1, (1,c0+1): 1},
            {(c0,1): 1}
        )
        
        # Now we need to combine these into C.
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        # My add_term_flex adds the terms to U, V, W directly.
        # But the W coefficients depend on the combination.
        # In the above, I put W coefficients as 1.
        # This is WRONG.
        # The W coefficients must reflect the combination.
        # For M1, it contributes to Z11 (coeff 1) and Z22 (coeff 1).
        # So W for M1 should be {(c0,0): 1, (c0+1,1): 1}.
        # For M4, it contributes to Z11 (coeff 1) and Z21 (coeff 1).
        # So W for M4 should be {(c0,0): 1, (c0,1): 1}.
        # Wait, Z21 is C10. Index (c0, 1)?
        # C is 3x8. Z21 is the bottom-left 1x1 of the 2x2 result?
        # No, the 2x2 result is for rows 0,1 and cols c0, c0+1.
        # Z11 is C[0, c0]. Z12 is C[0, c0+1].
        # Z21 is C[1, c0]. Z22 is C[1, c0+1].
        # So:
        # M1: W {(c0,0): 1, (c0+1,1): 1}
        # M2: W {(c0,1): 1}
        # M3: W {(c0+1,0): 1}
        # M4: W {(c0,0): 1, (c0,1): 1} ? No.
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # Let's re-assign W in the calls above.
        # I will clear U, V, W and redo.

    U.clear(); V.clear(); W.clear()

    def strassen_2x2_v2(c0):
        # M1
        add_term_flex(
            {(0,0): 1, (1,1): 1},
            {(0,c0): 1, (1,c0+1): 1},
            {(c0,0): 1, (c0+1,1): 1}
        )
        # M2
        add_term_flex(
            {(1,0): 1, (1,1): 1},
            {(0,c0): 1},
            {(c0,1): 1}
        )
        # M3
        add_term_flex(
            {(0,0): 1},
            {(0,c0+1): 1, (1,c0+1): -1},
            {(c0+1,0): 1}
        )
        # M4
        add_term_flex(
            {(1,1): 1},
            {(1,c0): 1, (0,c0): -1},
            {(c0,0): 1, (c0,1): 1}
        )
        # M5
        add_term_flex(
            {(0,0): 1, (0,1): 1},
            {(1,c0+1): 1},
            {(c0+1,0): -1, (c0+1,1): 1}
        )
        # M6
        add_term_flex(
            {(1,0): 1, (0,0): -1},
            {(0,c0): 1, (0,c0+1): 1},
            {(c0+1,1): 1}
        )
        # M7
        add_term_flex(
            {(0,1): 1, (1,1): -1},
            {(1,c0): 1, (1,c0+1): 1},
            {(c0,0): 1, (c0,1): -1}
        )

    # Apply Strassen to B11a (c0=0) and B11b (c0=2)
    strassen_2x2_v2(0)
    strassen_2x2_v2(2)
    # Apply Strassen to B12a (c0=4) and B12b (c0=6)
    strassen_2x2_v2(4)
    strassen_2x2_v2(6)

    # Now handle the remaining terms
    # A12 B21: A[0:2, 2] * B[2, 0:4]
    for i in range(2):
        for k in range(4):
            add_term_flex(
                {(i, 2): 1},
                {(2, k): 1},
                {(k, i): 1}
            )

    # A12 B22: A[0:2, 2] * B[2, 4:8]
    for i in range(2):
        for k in range(4, 8):
            add_term_flex(
                {(i, 2): 1},
                {(2, k): 1},
                {(k, i): 1}
            )

    # A21 B11: A[2, 0:2] * B[0:2, 0:4]
    for j in range(2):
        for k in range(4):
            add_term_flex(
                {(2, j): 1},
                {(j, k): 1},
                {(k, 2): 1}
            )

    # A21 B12: A[2, 0:2] * B[0:2, 4:8]
    for j in range(2):
        for k in range(4, 8):
            add_term_flex(
                {(2, j): 1},
                {(j, k): 1},
                {(k, 2): 1}
            )

    # A22 B21: A[2, 2] * B[2, 0:4]
    for k in range(4):
        add_term_flex(
            {(2, 2): 1},
            {(2, k): 1},
            {(k, 2): 1}
        )

    # A22 B22: A[2, 2] * B[2, 4:8]
    for k in range(4, 8):
        add_term_flex(
            {(2, 2): 1},
            {(2, k): 1},
            {(k, 2): 1}
        )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
