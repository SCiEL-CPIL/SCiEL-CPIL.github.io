import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # We will accumulate the rank
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank
        U_rows.append([0] * rank + [coeff]) # This is inefficient, let's use lists of lists and extend
        # Better: maintain lists of length 'rank' and append
        pass

    # Let's use a list of tuples (u, v, w, coeff) and then convert to matrices
    terms = []
    
    def add_rank1(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    # Helper for Strassen 2x2
    # u_base: index of A[0,0] in U
    # v_base: index of B[0,0] in V
    # w_base: index of C[0,0] in W
    # A is 2x2, B is 2x2, C is 2x2
    # U indices: (0,0), (0,1), (1,0), (1,1) relative to u_base
    # V indices: (0,0), (0,1), (1,0), (1,1) relative to v_base
    # W indices: (0,0), (0,1), (1,0), (1,1) relative to w_base
    
    def strassen_2x2(u_base, v_base, w_base):
        # A11, A12, A21, A22
        a11 = u_base + 0
        a12 = u_base + 1
        a21 = u_base + 2
        a22 = u_base + 3
        
        # B11, B12, B21, B22
        b11 = v_base + 0
        b12 = v_base + 1
        b21 = v_base + 2
        b22 = v_base + 3
        
        # C11, C12, C21, C22
        c11 = w_base + 0
        c12 = w_base + 1
        c21 = w_base + 2
        c22 = w_base + 3
        
        # M1 = (A11+A22)(B11+B22)
        # We can't just add terms directly because of the sums.
        # We need to decompose the bilinear form.
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is 4 terms. But Strassen uses 7 multiplications total.
        # The "multiplications" are the rank-1 terms in the tensor decomposition.
        # In the tensor decomposition, we define 7 new variables (intermediates).
        # But here we are constructing the U, V, W matrices directly.
        # The rank is the number of columns.
        # So we add 7 columns.
        
        # Col 0: M1
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22
        # This is not a rank-1 term. A rank-1 term has ONE non-zero in U, ONE in V, ONE in W.
        # So we must expand the Strassen algorithm into rank-1 terms?
        # No, Strassen's algorithm IS a rank-7 decomposition.
        # It uses 7 "multiplications" $M_k$.
        # $M_k = u_k^T A v_k$? No.
        # $M_k = (\text{linear combo of A}) \times (\text{linear combo of B})$.
        # This is a rank-1 term in the tensor sense?
        # Yes, $M_k$ corresponds to a rank-1 tensor $U_k \otimes V_k \otimes W_k$.
        # $U_k$ is the vector of coefficients for A.
        # $V_k$ is the vector of coefficients for B.
        # $W_k$ is the vector of coefficients for C (how $M_k$ contributes to C).
        
        # So for M1 = (A11+A22)(B11+B22):
        # U vector: 1 at A11, 1 at A22.
        # V vector: 1 at B11, 1 at B22.
        # W vector: 1 at C11, 1 at C22.
        # This is ONE rank-1 term.
        
        # So we add 1 column to U, V, W for each M_k.
        
        # M1
        add_rank1([a11, a22], [b11, b22], [c11, c22])
        
        # M2 = (A21+A22)B11
        add_rank1([a21, a22], [b11], [c21])
        
        # M3 = A11(B12-B22)
        add_rank1([a11], [b12, b22], [c12]) # Coeffs: 1, -1
        # Wait, add_rank1 takes indices. It assumes coeff 1.
        # I need to handle coefficients.
        # Let's modify add_rank1 to take lists of (idx, coeff).
        
        # Let's restart the helper logic.
        
    # Refined helper
    def add_strassen_2x2(u_base, v_base, w_base):
        a11 = u_base
        a12 = u_base + 1
        a21 = u_base + 2
        a22 = u_base + 3
        
        b11 = v_base
        b12 = v_base + 1
        b21 = v_base + 2
        b22 = v_base + 3
        
        c11 = w_base
        c12 = w_base + 1
        c21 = w_base + 2
        c22 = w_base + 3
        
        # M1
        terms.append(([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)]))
        # M2
        terms.append(([(a21,1),(a22,1)], [(b11,1)], [(c21,1)]))
        # M3
        terms.append(([(a11,1)], [(b12,1),(b22,-1)], [(c12,1)]))
        # M4
        terms.append(([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)]))
        # M5
        terms.append(([(a11,1),(a12,1)], [(b22,1)], [(c11,-1),(c12,1)]))
        # M6
        terms.append(([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)]))
        # M7
        terms.append(([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1),(c22,1)]))

    # Helper for 2x2 by 2x1 (Rank 4)
    def add_2x2_2x1(u_base, v_base, w_base):
        # A is 2x2, B is 2x1, C is 2x1
        a11 = u_base
        a12 = u_base + 1
        a21 = u_base + 2
        a22 = u_base + 3
        
        b11 = v_base
        b21 = v_base + 1
        
        c11 = w_base
        c21 = w_base + 1
        
        # C11 = A11B11 + A12B21
        terms.append(([(a11,1)], [(b11,1)], [(c11,1)]))
        terms.append(([(a12,1)], [(b21,1)], [(c11,1)]))
        # C21 = A21B11 + A22B21
        terms.append(([(a21,1)], [(b11,1)], [(c21,1)]))
        terms.append(([(a22,1)], [(b22,1)], [(c21,1)])) # Wait, a22 is index 3. b21 is index 1.
        # Correction:
        terms[-1] = ([(a22,1)], [(b21,1)], [(c21,1)])

    # Helper for 2x3 by 3x3 (Rank 17)
    def add_2x3_3x3(u_base, v_base, w_base):
        # A is 2x3. Indices: 0,1,2 (row 0), 3,4,5 (row 1)
        # B is 3x3. Indices: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
        # C is 2x3. Indices: 0,1,2 (row 0), 3,4,5 (row 1)
        
        # Part 1: A_col1 (2x1) * B_row1 (1x3). Rank 6.
        # A_col1: a11, a21 (indices 0, 3)
        # B_row1: b11, b12, b13 (indices 0, 1, 2)
        # C: c11, c12, c13 (indices 0, 1, 2) and c21, c22, c23 (indices 3, 4, 5)
        
        # c11 = a11 b11
        terms.append(([(u_base+0, 1)], [(v_base+0, 1)], [(w_base+0, 1)]))
        # c12 = a11 b12
        terms.append(([(u_base+0, 1)], [(v_base+1, 1)], [(w_base+1, 1)]))
        # c13 = a11 b13
        terms.append(([(u_base+0, 1)], [(v_base+2, 1)], [(w_base+2, 1)]))
        # c21 = a21 b11
        terms.append(([(u_base+3, 1)], [(v_base+0, 1)], [(w_base+3, 1)]))
        # c22 = a21 b12
        terms.append(([(u_base+3, 1)], [(v_base+1, 1)], [(w_base+4, 1)]))
        # c23 = a21 b13
        terms.append(([(u_base+3, 1)], [(v_base+2, 1)], [(w_base+5, 1)]))
        
        # Part 2: A_rest (2x2) * B_rest (2x3). Rank 11.
        # A_rest: a12, a13 (indices 1, 2), a22, a23 (indices 4, 5)
        # B_rest: b21, b22, b23 (indices 3, 4, 5), b31, b32, b33 (indices 6, 7, 8)
        # B_rest is 2x3.
        # Split B_rest into B_2x2 (cols 0,1) and B_2x1 (col 2).
        # B_2x2: b21, b22, b31, b32 (indices 3, 4, 6, 7)
        # B_2x1: b23, b33 (indices 5, 8)
        
        # Part 2a: 2x2 by 2x2 (Strassen). Rank 7.
        # u_base_2a = u_base + 1 (a12)
        # v_base_2a = v_base + 3 (b21)
        # w_base_2a = w_base + 0 (c11)
        # Note: The result of this part adds to C.
        # C indices for this part:
        # c11, c12, c21, c22.
        # In global C indices: 0, 1, 3, 4.
        
        # We need a specialized Strassen that writes to specific W indices.
        def add_strassen_2x2_custom(u_indices, v_indices, w_indices):
            # u_indices: [a11, a12, a21, a22]
            # v_indices: [b11, b12, b21, b22]
            # w_indices: [c11, c12, c21, c22]
            
            a11, a12, a21, a22 = u_indices
            b11, b12, b21, b22 = v_indices
            c11, c12, c21, c22 = w_indices
            
            terms.append(([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)]))
            terms.append(([(a21,1),(a22,1)], [(b11,1)], [(c21,1)]))
            terms.append(([(a11,1)], [(b12,1),(b22,-1)], [(c12,1)]))
            terms.append(([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)]))
            terms.append(([(a11,1),(a12,1)], [(b22,1)], [(c11,-1),(c12,1)]))
            terms.append(([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)]))
            terms.append(([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1),(c22,1)]))

        # Call for Part 2a
        add_strassen_2x2_custom(
            [u_base+1, u_base+2, u_base+4, u_base+5],
            [v_base+3, v_base+4, v_base+6, v_base+7],
            [w_base+0, w_base+1, w_base+3, w_base+4]
        )
        
        # Part 2b: 2x2 by 2x1 (Rank 4).
        # A_rest: same as above.
        # B_2x1: b23, b33 (indices 5, 8).
        # C: c13, c23 (indices 2, 5).
        
        def add_2x2_2x1_custom(u_indices, v_indices, w_indices):
            a11, a12, a21, a22 = u_indices
            b11, b21 = v_indices
            c11, c21 = w_indices
            
            terms.append(([(a11,1)], [(b11,1)], [(c11,1)]))
            terms.append(([(a12,1)], [(b21,1)], [(c11,1)]))
            terms.append(([(a21,1)], [(b11,1)], [(c21,1)]))
            terms.append(([(a22,1)], [(b21,1)], [(c21,1)]))

        add_2x2_2x1_custom(
            [u_base+1, u_base+2, u_base+4, u_base+5],
            [v_base+5, v_base+8],
            [w_base+2, w_base+5]
        )

    # --- Main Construction ---
    
    # Term 1: A1 (2x2) * B1 (2x6)
    # A1: rows 0,1 cols 0,1. U indices: 0,1,2,3.
    # B1: rows 0,1 cols 0..5. V indices: 0..5 (row 0), 6..11 (row 1).
    # B1a: cols 0,1. V: 0,1,6,7.
    # B1b: cols 2,3. V: 2,3,8,9.
    # B1c: cols 4,5. V: 4,5,10,11.
    # C1: rows 0,1 cols 0..5. W: 0..1 (col 0), 2..3 (col 1), 4..5 (col 2), 6..7 (col 3), 8..9 (col 4), 10..11 (col 5).
    # W indices for B1a: 0,1,2,3.
    # W indices for B1b: 4,5,6,7.
    # W indices for B1c: 8,9,10,11.
    
    add_strassen_2x2_custom(
        [0, 1, 2, 3],
        [0, 1, 6, 7],
        [0, 1, 2, 3]
    )
    add_strassen_2x2_custom(
        [0, 1, 2, 3],
        [2, 3, 8, 9],
        [4, 5, 6, 7]
    )
    add_strassen_2x2_custom(
        [0, 1, 2, 3],
        [4, 5, 10, 11],
        [8, 9, 10, 11]
    )
    
    # Term 2: A2 (2x3) * B2 (3x6)
    # A2: rows 0,1 cols 2,3,4. U indices: 4,5,6 (row 0), 7,8,9 (row 1).
    # B2: rows 2,3,4 cols 0..5.
    # V indices for B2:
    # Row 2 (global row 2): 12..17
    # Row 3 (global row 3): 18..23
    # Row 4 (global row 4): 24..29
    # B2a: cols 0,1,2. V: 12,13,14 (row 2), 18,19,20 (row 3), 24,25,26 (row 4).
    # B2b: cols 3,4,5. V: 15,16,17 (row 2), 21,22,23 (row 3), 27,28,29 (row 4).
    
    # C2: rows 0,1 cols 0..5.
    # W indices for B2a: 0,1,2 (col 0), 2..3? No.
    # W is (p*n, rank) = (6*2, rank) = (12, rank).
    # W index = k*n + i. k is col of C, i is row of C.
    # B2a produces cols 0,1,2 of C.
    # Col 0: k=0. W: 0,1.
    # Col 1: k=1. W: 2,3.
    # Col 2: k=2. W: 4,5.
    # So W indices for B2a: 0,1, 2,3, 4,5.
    
    # B2b produces cols 3,4,5 of C.
    # Col 3: k=3. W: 6,7.
    # Col 4: k=4. W: 8,9.
    # Col 5: k=5. W: 10,11.
    # So W indices for B2b: 6,7, 8,9, 10,11.
    
    add_2x3_3x3(
        [4, 5, 6, 7, 8, 9], # U indices for A2
        [12, 13, 14, 18, 19, 20, 24, 25, 26], # V indices for B2a
        [0, 1, 2, 3, 4, 5] # W indices for C2a
    )
    
    add_2x3_3x3(
        [4, 5, 6, 7, 8, 9], # U indices for A2
        [15, 16, 17, 21, 22, 23, 27, 28, 29], # V indices for B2b
        [6, 7, 8, 9, 10, 11] # W indices for C2b
    )
    
    # Build matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U[idx][c] += coeff
        for idx, coeff in v_terms:
            V[idx][c] += coeff
        for idx, coeff in w_terms:
            W[idx][c] += coeff
            
    return {"U": U, "V": V, "W": W}

