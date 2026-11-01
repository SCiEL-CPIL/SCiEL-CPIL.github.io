"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1, A2 (2x2)
    # Split B into B11, B12, B21, B22 (2x2)
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # We compute 4 products:
    # P1 = A1*B11
    # P2 = A2*B21
    # P3 = A1*B12
    # P4 = A2*B22
    
    # Each 2x2 product has rank 7.
    # Total rank 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Indices for U (A): i*m + j
    # i in 0..1, j in 0..3
    # A1: i=0,1 j=0,1
    # A2: i=0,1 j=2,3
    
    # Indices for V (B): j*p + k
    # j in 0..3, k in 0..3
    # B11: j=0,1 k=0,1
    # B12: j=0,1 k=2,3
    # B21: j=2,3 k=0,1
    # B22: j=2,3 k=2,3
    
    # Indices for W (C): k*n + i
    # k in 0..3, i in 0..1
    # C1: k=0,1 (cols 0,1 of C)
    # C2: k=2,3 (cols 2,3 of C)
    # Note: W index is k*n + i.
    # C is 2x4. Row i, Col k.
    # W[k*n + i, r] corresponds to C[i, k].
    
    def add_strassen_product(u_base, v_base, w_base, offset):
        # u_base: (i, j) indices for the 2x2 block of A
        # v_base: (j, k) indices for the 2x2 block of B
        # w_base: (k, i) indices for the 2x2 block of C
        # offset: starting column in U, V, W
        
        # Strassen formulas for 2x2 multiplication
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to the global indices.
        # For a specific 2x2 block of A (rows r1,r2; cols c1,c2)
        # and B (rows r3,r4; cols c3,c4)
        # and C (rows r5,r6; cols c5,c6)
        
        # This function is too complex to generalize easily without careful index mapping.
        # Instead, we can just hardcode the 28 terms.
        pass

    # Hardcoded Rank 28 Decomposition
    # Term 1-7: A1*B11
    # A1 is A[0:2, 0:2]
    # B11 is B[0:2, 0:2]
    # C1_part1 is C[0:2, 0:2]
    
    # Let's use a helper to generate the 7 terms for a 2x2x2 multiplication
    # and map them to the global tensors.
    
    def get_strassen_terms(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # Returns list of (u_idx, v_idx, w_idx, coeff)
        # a_rows: [r1, r2], a_cols: [c1, c2]
        # b_rows: [r3, r4], b_cols: [c3, c4]
        # c_rows: [r5, r6], c_cols: [c5, c6]
        
        terms = []
        
        # M1
        # (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C12, C21, C22? No, W is the output.
        # The bilinear form is sum of products.
        # U vector has 1s at A11, A22.
        # V vector has 1s at B11, B22.
        # W vector has 1s at C11, C12, C21, C22?
        # No, the result of M1 is added to C11, C12, C21, C22.
        # Specifically, C11 += M1, C12 += M1, C21 += M1, C22 += M1?
        # No.
        # C11 = M1 + M4 - M2 - M5
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        
        # So for M1, it contributes +1 to C11, -1 to C22? No.
        # M1 contributes +1 to C11, +1 to C22.
        # Wait, C22 = M1 - ... so +1 to C22.
        
        # Let's list the contributions of each M to C blocks.
        # M1: C11 +1, C22 +1
        # M2: C11 -1, C21 +1
        # M3: C12 +1, C22 -1
        # M4: C11 +1, C21 +1
        # M5: C11 -1, C12 +1
        # M6: C22 +1
        # M7: C22 +1
        
        # This mapping is for the local 2x2 C.
        
        # We need to map local (i,j) to global W index.
        # Local i=0,1; j=0,1.
        # Global W index = c_cols[j]*n + c_rows[i]
        
        # For M1:
        # U: A11 (a_rows[0], a_cols[0]), A22 (a_rows[1], a_cols[1])
        # V: B11 (b_rows[0], b_cols[0]), B22 (b_rows[1], b_cols[1])
        # W: C11 (c_rows[0], c_cols[0]) coeff 1, C22 (c_rows[1], c_cols[1]) coeff 1
        
        # We will return 7 entries for the 7 M's.
        
        # M1
        u1 = [a_rows[0]*m + a_cols[0], a_rows[1]*m + a_cols[1]]
        v1 = [b_rows[0]*p + b_cols[0], b_rows[1]*p + b_cols[1]]
        w1 = [(c_cols[0]*n + c_rows[0], 1), (c_cols[1]*n + c_rows[1], 1)]
        
        # M2
        u2 = [a_rows[1]*m + a_cols[0], a_rows[1]*m + a_cols[1]]
        v2 = [b_rows[0]*p + b_cols[0]]
        w2 = [(c_cols[0]*n + c_rows[0], -1), (c_cols[0]*n + c_rows[1], 1)]
        
        # M3
        u3 = [a_rows[0]*m + a_cols[0]]
        v3 = [b_rows[0]*p + b_cols[1], b_rows[1]*p + b_cols[1]]
        w3 = [(c_cols[1]*n + c_rows[0], 1), (c_cols[1]*n + c_rows[1], -1)]
        
        # M4
        u4 = [a_rows[1]*m + a_cols[1]]
        v4 = [b_rows[1]*p + b_cols[0], b_rows[0]*p + b_cols[0]]
        w4 = [(c_cols[0]*n + c_rows[0], 1), (c_cols[0]*n + c_rows[1], 1)]
        
        # M5
        u5 = [a_rows[0]*m + a_cols[0], a_rows[0]*m + a_cols[1]]
        v5 = [b_rows[1]*p + b_cols[1]]
        w5 = [(c_cols[0]*n + c_rows[0], -1), (c_cols[1]*n + c_rows[0], 1)]
        
        # M6
        u6 = [a_rows[1]*m + a_cols[0], a_rows[0]*m + a_cols[0]]
        v6 = [b_rows[0]*p + b_cols[0], b_rows[0]*p + b_cols[1]]
        w6 = [(c_cols[1]*n + c_rows[1], 1)]
        
        # M7
        u7 = [a_rows[0]*m + a_cols[1], a_rows[1]*m + a_cols[1]]
        v7 = [b_rows[1]*p + b_cols[0], b_rows[1]*p + b_cols[1]]
        w7 = [(c_cols[1]*n + c_rows[1], 1)]
        
        # Note: U and V entries are indices. Coefficients are 1 or -1.
        # We need to store these in the global U, V, W matrices.
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]

    # Define the 4 blocks
    # A1: rows 0,1; cols 0,1
    # A2: rows 0,1; cols 2,3
    # B11: rows 0,1; cols 0,1
    # B12: rows 0,1; cols 2,3
    # B21: rows 2,3; cols 0,1
    # B22: rows 2,3; cols 2,3
    # C1: rows 0,1; cols 0,1
    # C2: rows 0,1; cols 2,3
    
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]), # A1*B11 -> C1
        ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1]), # A2*B21 -> C1
        ([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3]), # A1*B12 -> C2
        ([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3]), # A2*B22 -> C2
    ]
    
    col = 0
    for b in blocks:
        terms = get_strassen_terms(*b)
        for u_indices, v_indices, w_terms in terms:
            for idx in u_indices:
                U[idx][col] = 1
            for idx in v_indices:
                V[idx][col] = 1
            for w_idx, coeff in w_terms:
                W[w_idx][col] = coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
