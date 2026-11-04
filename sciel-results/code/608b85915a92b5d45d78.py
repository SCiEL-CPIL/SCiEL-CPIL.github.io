"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 decomposition of the 2x3x8 matrix multiplication tensor.
    
    Strategy: Split the 2x3x8 problem into two sub-problems:
    - 2x2x8 block (rows 0-1 of first matrix, cols 0-1 of second)
    - 2x1x8 block (rows 0-1 of first matrix, col 2 of second)
    
    For the 2x2x8 block, use Strassen's algorithm (rank 7) for the 2x2x2 part
    combined with the 8 columns. Actually, let me think more carefully.
    
    The 2x3x8 multiplication: C = A * B where A is 2x3, B is 3x8, C is 2x8.
    
    We can decompose this as:
    C[:, 0:8] = A[:, 0:2] @ B[0:2, 0:8] + A[:, 2:3] @ B[2:3, 0:8]
    
    So it's a direct sum of:
    - 2x2x8 problem: A1 (2x2) times B1 (2x8)
    - 2x1x8 problem: A2 (2x1) times B2 (1x8)
    
    For the 2x1x8 problem, the rank is 8 (each output column needs one rank-1 term).
    For the 2x2x8 problem, we can use Strassen-like decomposition.
    
    Actually, let me reconsider. The standard approach:
    
    For 2x2x2, Strassen gives rank 7.
    For 2x2x8, we can think of it as 2x2 matrix times 8 columns.
    This is equivalent to 4 separate 2x1x1 multiplications if done naively (rank 8),
    but we can do better by grouping.
    
    Actually, 2x2x8 can be viewed as: for each of the 8 columns of B, we compute
    a 2x1 product. But we can share structure.
    
    Let me think differently. The 2x2x8 tensor has:
    - Input A: 2x2 = 4 entries
    - Input B: 2x8 = 16 entries  
    - Output C: 2x8 = 16 entries
    
    Rank of 2x2xn matrix multiplication is known. For n=8, we can use:
    - Decompose into 2x2x2 blocks? No, that doesn't work directly.
    
    Better approach: Use the fact that 2x2x8 = (2x2x2) ⊗ (1x1x4) in some sense?
    No, that's not quite right either.
    
    Let me just use a direct construction:
    
    For the 2x1x8 part: rank 8, trivial.
    For the 2x2x8 part: 
    
    We can write 2x2x8 as a direct sum of four 2x1x1 problems, but that gives rank 8.
    However, we can do better.
    
    Actually, the rank of M_{2,2,n} is known to be:
    - For n=1: rank 4 (but Strassen says 7 for 2x2x2, so this is wrong)
    
    Wait, I'm confusing myself. Let me restart.
    
    M_{n,m,p} is the tensor for multiplying an n x m matrix by an m x p matrix.
    Rank(M_{2,2,2}) = 7 (Strassen).
    
    For M_{2,2,8}, we can think of it as:
    C_{i,k} = sum_j A_{i,j} B_{j,k} for i=0,1 and k=0,...,7
    
    This is 16 outputs, each a bilinear form.
    
    One approach: Group the 8 columns of B into 4 pairs. For each pair (k, k+2),
    we have a 2x2x2 sub-problem (A is 2x2, B_pair is 2x2, C_pair is 2x2).
    Using Strassen for each: 4 * 7 = 28.
    
    Plus the 2x1x8 part: rank 8.
    Total: 36.
    
    But wait, that's not quite right because the A matrix is shared.
    
    Actually, the direct sum structure means:
    M_{2,3,8} = M_{2,2,8} ⊕ M_{2,1,8}
    rank(M_{2,3,8}) = rank(M_{2,2,8}) + rank(M_{2,1,8})
    
    rank(M_{2,1,8}) = 8 (trivial, each output is A_{i,2} * B_{2,k})
    
    For M_{2,2,8}:
    We can decompose B (2x8) into 4 blocks of 2x2.
    Then C = A * B = A * [B_0, B_1, B_2, B_3] where B_i is 2x2.
    C_i = A * B_i for i=0,1,2,3, each a 2x2x2 problem.
    So M_{2,2,8} = M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,2}
    rank(M_{2,2,8}) = 4 * rank(M_{2,2,2}) = 4 * 7 = 28.
    
    Total rank: 28 + 8 = 36.
    
    Let me implement this.
    """
    import time
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    
    n, m, p = 2, 3, 8
    
    # We'll build the decomposition in two parts:
    # Part 1: 2x2x8 (using 4 Strassen 2x2x2 decompositions)
    # Part 2: 2x1x8 (trivial, rank 8)
    
    # Strassen decomposition for 2x2x2:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # Strassen's 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to express it as rank-1 contributions.
    # Each product like (linear_comb_of_A)*(linear_comb_of_B) is rank 1 in the tensor.
    
    def strassen_decomp(offset_U, offset_V, offset_W, a_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for a 2x2x2 sub-problem.
        a_rows: indices of A rows (2 entries)
        b_cols: indices of B cols (2 entries)  
        c_rows: indices of C rows (2 entries)
        c_cols: indices of C cols (2 entries)
        Returns new offset.
        """
        U_rows = []
        V_rows = []
        W_rows = []
        
        # We'll build U, V, W entries for this sub-problem
        # U has shape (n*m, rank), so for A part, row index is i*m + j
        # V has shape (m*p, rank), so for B part, row index is j*p + k
        # W has shape (p*n, rank), so for C part, row index is k*n + i
        
        # For the 2x2x2 sub-problem with specific row/col indices:
        # A indices: (a_rows[0], b_rows[0]), (a_rows[0], b_rows[1]), 
        #            (a_rows[1], b_rows[0]), (a_rows[1], b_rows[1])
        # Wait, I need to be more careful about the indexing.
        
        # Let me redefine: for the 2x2x8 part, we have:
        # A sub-matrix: rows 0,1 of A, cols 0,1 of A (so A[0:2, 0:2])
        # B sub-matrix: rows 0,1 of B, cols c_cols[0], c_cols[1] of B
        # C sub-matrix: rows 0,1 of C, cols c_cols[0], c_cols[1] of C
        
        # In the full problem:
        # A[i,j] is at U row i*m + j = i*3 + j
        # B[j,k] is at V row j*p + k = j*8 + k
        # C[i,k] is at W row k*n + i = k*2 + i
        
        # For this sub-problem, the relevant A entries are at rows:
        # A[0,0] -> U row 0*3+0 = 0
        # A[0,1] -> U row 0*3+1 = 1
        # A[1,0] -> U row 1*3+0 = 3
        # A[1,1] -> U row 1*3+1 = 4
        
        # B entries at rows:
        # B[0, c_cols[0]] -> V row 0*8 + c_cols[0]
        # B[0, c_cols[1]] -> V row 0*8 + c_cols[1]
        # B[1, c_cols[0]] -> V row 1*8 + c_cols[0]
        # B[1, c_cols[1]] -> V row 1*8 + c_cols[1]
        
        # C entries at rows:
        # C[0, c_cols[0]] -> W row c_cols[0]*2 + 0
        # C[0, c_cols[1]] -> W row c_cols[1]*2 + 0
        # C[1, c_cols[0]] -> W row c_cols[0]*2 + 1
        # C[1, c_cols[1]] -> W row c_cols[1]*2 + 1
        
        # Strassen products for this sub-problem:
        # Let a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
        # Let b00=B[0,c0], b01=B[0,c1], b10=B[1,c0], b11=B[1,c1] where c0=c_cols[0], c1=c_cols[1]
        
        c0, c1 = c_cols
        
        # Define the 7 Strassen linear forms
        # Each is (coeffs for A, coeffs for B)
        strassen_terms = [
            # (A_coeffs, B_coeffs) where coeffs are dicts mapping (i,j) or (j,k) to coefficient
            # M1 = (a00+a11)*(b00+b11)
            ({(0,0): 1, (1,1): 1}, {(0,c0): 1, (1,c1): 1}),
            # M2 = (a10+a11)*b00
            ({(1,0): 1, (1,1): 1}, {(0,c0): 1}),
            # M3 = a00*(b01-b11)
            ({(0,0): 1}, {(0,c1): 1, (1,c1): -1}),
            # M4 = a11*(b10-b00)
            ({(1,1): 1}, {(1,c0): 1, (0,c0): -1}),
            # M5 = (a00+a01)*b11
            ({(0,0): 1, (0,1): 1}, {(1,c1): 1}),
            # M6 = (a10-a00)*(b00+b10)
            ({(1,0): 1, (0,0): -1}, {(0,c0): 1, (1,c0): 1}),
            # M7 = (a01-a11)*(b01+b11)
            ({(0,1): 1, (1,1): -1}, {(0,c1): 1, (1,c1): 1}),
        ]
        
        # Output reconstruction:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # C[0,c0] = M1 + M4 - M5 + M7
        # C[0,c1] = M3 + M5
        # C[1,c0] = M2 + M4
        # C[1,c1] = M1 - M2 + M3 + M6
        
        output_coeffs = [
            # (c_row, c_col, {m_index: coeff})
            (0, c0, {0: 1, 3: 1, 4: -1, 6: 1}),   # C[0,c0]
            (0, c1, {2: 1, 4: 1}),                 # C[0,c1]
            (1, c0, {1: 1, 3: 1}),                 # C[1,c0]
            (1, c1, {0: 1, 1: -1, 2: 1, 5: 1}),   # C[1,c1]
        ]
        
        new_offset = offset_U
        for m_idx in range(7):
            A_coeffs, B_coeffs = strassen_terms[m_idx]
            # Add to U
            for (i, j), coeff in A_coeffs.items():
                U_row = i * m + j
                # We'll store these and add later
                pass
            
            # Add to V
            for (j, k), coeff in B_coeffs.items():
                V_row = j * p + k
                pass
        
        # Actually, let me build the full U, V, W matrices directly.
        # This function approach is getting complicated. Let me just build everything in one go.
        
        return new_offset
    
    # Let me take a cleaner approach and build the full decomposition directly.
    
    # Total rank: 4 (for 2x2x8, using 4 Strassen) + 8 (for 2x1x8) = 36
    # Wait, 4 Strassen decompositions each have rank 7, so 4*7 = 28.
    # Plus 8 for the 2x1x8 part. Total = 36.
    
    total_rank = 4 * 7 + 8  # = 36
    
    U = [[0] * total_rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 3*8 = 24 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    col = 0
    
    # Part 1: 2x2x8 using 4 Strassen decompositions
    # For each pair of columns (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        c0 = block * 2
        c1 = block * 2 + 1
        
        # Strassen terms for this 2x2x2 sub-problem
        strassen_terms = [
            # (A_coeffs: {(i,j): coeff}, B_coeffs: {(j,k): coeff})
            ({(0,0): 1, (1,1): 1}, {(0,c0): 1, (1,c1): 1}),  # M1
            ({(1,0): 1, (1,1): 1}, {(0,c0): 1}),              # M2
            ({(0,0): 1}, {(0,c1): 1, (1,c1): -1}),            # M3
            ({(1,1): 1}, {(1,c0): 1, (0,c0): -1}),            # M4
            ({(0,0): 1, (0,1): 1}, {(1,c1): 1}),              # M5
            ({(1,0): 1, (0,0): -1}, {(0,c0): 1, (1,c0): 1}),  # M6
            ({(0,1): 1, (1,1): -1}, {(0,c1): 1, (1,c1): 1}),  # M7
        ]
        
        # Output reconstruction coefficients
        # C[i,k] = sum_m coeff_m * M_m
        output_coeffs = [
            # (i, k, {m_idx: coeff})
            (0, c0, {0: 1, 3: 1, 4: -1, 6: 1}),   # C[0,c0] = M1 + M4 - M5 + M7
            (0, c1, {2: 1, 4: 1}),                 # C[0,c1] = M3 + M5
            (1, c0, {1: 1, 3: 1}),                 # C[1,c0] = M2 + M4
            (1, c1, {0: 1, 1: -1, 2: 1, 5: 1}),   # C[1,c1] = M1 - M2 + M3 + M6
        ]
        
        for m_idx in range(7):
            A_coeffs, B_coeffs = strassen_terms[m_idx]
            
            # Add to U (A part)
            for (i, j), coeff in A_coeffs.items():
                U_row = i * m + j
                U[U_row][col] += coeff
            
            # Add to V (B part)
            for (j, k), coeff in B_coeffs.items():
                V_row = j * p + k
                V[V_row][col] += coeff
            
            # Add to W (C part)
            for (i, k, m_coeff_dict) in output_coeffs:
                if m_idx in m_coeff_dict:
                    W_row = k * n + i
                    W[W_row][col] += m_coeff_dict[m_idx]
            
            col += 1
        
        if time.time() > deadline:
            break
    
    # Part 2: 2x1x8 (trivial)
    # C[i,k] = A[i,2] * B[2,k] for i=0,1 and k=0,...,7
    for k in range(8):
        if time.time() > deadline:
            break
        # A[0,2] * B[2,k] -> C[0,k]
        U[0 * m + 2][col] = 1  # A[0,2]
        V[2 * p + k][col] = 1  # B[2,k]
        W[k * n + 0][col] = 1  # C[0,k]
        col += 1
        
        # A[1,2] * B[2,k] -> C[1,k]
        U[1 * m + 2][col] = 1  # A[1,2]
        V[2 * p + k][col] = 1  # B[2,k]
        W[k * n + 1][col] = 1  # C[1,k]
        col += 1
    
    # Trim to actual used columns
    if col < total_rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
