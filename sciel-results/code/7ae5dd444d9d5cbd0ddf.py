"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a 3+2 split for the 5x5 matrices.
    # C = [[C11, C12], [C21, C22]]
    # C11 is 3x3, C12 is 3x2, C21 is 2x3, C22 is 2x2
    
    # We will construct U, V, W for the whole 5x5 multiplication.
    # Rank = Rank(C11) + Rank(C12) + Rank(C21) + Rank(C22)
    
    # C11: 3x3x3 (schoolbook 27) + 3x2x3 (schoolbook 18) = 45
    # C12: 3x3x3 (schoolbook 27) + 3x2x2 (schoolbook 12) = 39? 
    # Wait, schoolbook for 3x3x2 is 3*3*2 = 18? No.
    # C12 is 3x2. C12[i,j] = sum_k A[11,i,k] B[12,k,j] + sum_k A[12,i,k] B[22,k,j]
    # A11 is 3x3, B12 is 3x2. Product is 3x2. Rank 3*3*2 = 18.
    # A12 is 3x2, B22 is 2x2. Product is 3x2. Rank 3*2*2 = 12.
    # Total 30.
    
    # C21: 2x3x3 (18) + 2x2x3 (12) = 30.
    
    # C22: 2x3x2 (12) + 2x2x2 (Strassen 7) = 19.
    
    # Total Rank = 45 + 30 + 30 + 19 = 124.
    
    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(n_block, m_block, p_block, i_offset, j_offset, k_offset):
        nonlocal col
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    u_idx = (i_offset + i) * m + (j_offset + j)
                    v_idx = (j_offset + j) * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + (i_offset + i)
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    # Helper to add Strassen 2x2 block
    def add_strassen(n_block, m_block, p_block, i_offset, j_offset, k_offset):
        # Strassen for 2x2
        # A11 A12
        # A21 A22
        # B11 B12
        # B21 B22
        #
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices
        # A11: i_offset, j_offset
        # A12: i_offset, j_offset+1
        # A21: i_offset+1, j_offset
        # A22: i_offset+1, j_offset+1
        
        # We need to add 7 terms.
        # Each term is a product of a linear combination of A and a linear combination of B.
        # We will expand these into the U, V, W matrices.
        
        # M1: (A11+A22)(B11+B22)
        # A part: A11 + A22
        # B part: B11 + B22
        # This is 4 rank-1 terms if we expand? No, it's 1 multiplication of sums.
        # In tensor decomposition, we add 1 term for each product of a U-vector and V-vector.
        # The U-vector is the sum of the basis vectors for A11 and A22.
        # The V-vector is the sum of the basis vectors for B11 and B22.
        
        # We define the terms as tuples of (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict mapping (i,j) to coeff
        # v_coeffs: dict mapping (j,k) to coeff
        # w_coeffs: dict mapping (k,i) to coeff
        
        terms = [
            # M1
            ({(i_offset, j_offset): 1, (i_offset+1, j_offset+1): 1}, 
             {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1},
             {(k_offset, i_offset): 1, (k_offset, i_offset+1): 1, (k_offset+1, i_offset): 1, (k_offset+1, i_offset+1): 1}),
            # M2
            ({(i_offset+1, j_offset): 1, (i_offset+1, j_offset+1): 1},
             {(j_offset, k_offset): 1},
             {(k_offset, i_offset+1): 1}),
            # M3
            ({(i_offset, j_offset): 1},
             {(j_offset, k_offset+1): 1, (j_offset+1, k_offset+1): -1},
             {(k_offset+1, i_offset): 1}),
            # M4
            ({(i_offset+1, j_offset+1): 1},
             {(j_offset+1, k_offset): 1, (j_offset, k_offset): -1},
             {(k_offset, i_offset+1): 1}),
            # M5
            ({(i_offset, j_offset): 1, (i_offset, j_offset+1): 1},
             {(j_offset+1, k_offset+1): 1},
             {(k_offset+1, i_offset): 1}),
            # M6
            ({(i_offset+1, j_offset): 1, (i_offset, j_offset): -1},
             {(j_offset, k_offset): 1, (j_offset, k_offset+1): 1},
             {(k_offset, i_offset+1): 1, (k_offset+1, i_offset+1): 1}),
            # M7
            ({(i_offset, j_offset+1): 1, (i_offset+1, j_offset+1): -1},
             {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): 1},
             {(k_offset, i_offset): 1, (k_offset+1, i_offset): 1}),
        ]
        
        for u_c, v_c, w_c in terms:
            for (i, j), coeff in u_c.items():
                U[i * m + j][col] += coeff
            for (j, k), coeff in v_c.items():
                V[j * p + k][col] += coeff
            for (k, i), coeff in w_c.items():
                W[k * n + i][col] += coeff
            col += 1

    # C11: 3x3 block at (0,0)
    # A11 (3x3) * B11 (3x3)
    add_schoolbook(3, 3, 3, 0, 0, 0)
    # A12 (3x2) * B21 (2x3)
    add_schoolbook(3, 2, 3, 0, 3, 3)
    
    # C12: 3x2 block at (0,3)
    # A11 (3x3) * B12 (3x2)
    add_schoolbook(3, 3, 2, 0, 0, 3)
    # A12 (3x2) * B22 (2x2)
    add_schoolbook(3, 2, 2, 0, 3, 3)
    
    # C21: 2x3 block at (3,0)
    # A21 (2x3) * B11 (3x3)
    add_schoolbook(2, 3, 3, 3, 0, 0)
    # A22 (2x2) * B21 (2x3)
    add_schoolbook(2, 2, 3, 3, 3, 3)
    
    # C22: 2x2 block at (3,3)
    # A21 (2x3) * B12 (3x2)
    add_schoolbook(2, 3, 2, 3, 0, 3)
    # A22 (2x2) * B22 (2x2)
    add_strassen(2, 2, 2, 3, 3, 3)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
