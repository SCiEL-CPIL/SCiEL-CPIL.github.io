"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    # Rank 66 decomposition
    rank = 66
    
    # U: (n*m, rank) = (12, 66)
    # V: (m*p, rank) = (24, 66)
    # W: (p*n, rank) = (18, 66)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Part 1: A_1 B (Row 0 of A times B)
    # A_1 is 1x4. B is 4x6.
    # C_1 is 1x6 (Row 0 of C).
    for j in range(m):
        for k in range(p):
            u_idx = 0 * m + j  # Row 0 of A, col j
            v_idx = j * p + k  # Row j of B, col k
            w_idx = k * n + 0  # Row 0 of C, col k
            add_term(u_idx, v_idx, w_idx, 1)
            
    # Part 2: A_2 B (Rows 1,2 of A times B)
    # A_2 is 2x4. Split into A_2L (cols 0,1) and A_2R (cols 2,3).
    # B is 4x6. Split into B_T (rows 0,1) and B_B (rows 2,3).
    # C_2 = A_2L B_T + A_2R B_B.
    
    # Strassen coefficients for 2x2x2
    # M = [[m00, m01], [m10, m11]]
    # P = [[p00, p01], [p10, p11]]
    # M1 = (m00+m11)(p00+p11)
    # M2 = (m10+m11)p00
    # M3 = m00(p01-p11)
    # M4 = m11(p10-p00)
    # M5 = (m00+m01)p11
    # M6 = (m10-m00)(p00+p01)
    # M7 = (m01-m11)(p10+p11)
    
    # Result R = [[r00, r01], [r10, r11]]
    # r00 = M1 + M4 - M5 + M7
    # r01 = M3 + M5
    # r10 = M2 + M4
    # r11 = M1 - M2 + M3 + M6
    
    # We need to map these linear combinations to the global U, V, W.
    # Each M_i is a product of a linear combo of A entries and a linear combo of B entries.
    # This corresponds to 7 rank-1 terms for the 2x2x2 block.
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        a_rows: list of 2 row indices for A (0..n-1)
        a_cols: list of 2 col indices for A (0..m-1)
        b_rows: list of 2 row indices for B (0..m-1)
        b_cols: list of 2 col indices for B (0..p-1)
        c_rows: list of 2 row indices for C (0..n-1)
        c_cols: list of 2 col indices for C (0..p-1)
        """
        # A indices
        u00 = a_rows[0] * m + a_cols[0]
        u01 = a_rows[0] * m + a_cols[1]
        u10 = a_rows[1] * m + a_cols[0]
        u11 = a_rows[1] * m + a_cols[1]
        
        # B indices
        v00 = b_rows[0] * p + b_cols[0]
        v01 = b_rows[0] * p + b_cols[1]
        v10 = b_rows[1] * p + b_cols[0]
        v11 = b_rows[1] * p + b_cols[1]
        
        # C indices
        w00 = c_cols[0] * n + c_rows[0]
        w01 = c_cols[1] * n + c_rows[0]
        w10 = c_cols[0] * n + c_rows[1]
        w11 = c_cols[1] * n + c_rows[1]
        
        # M1: (m00+m11)(p00+p11)
        # U: u00=1, u11=1. V: v00=1, v11=1. W: w00+=1, w11+=1.
        # We need to decompose this product of sums into rank 1 terms?
        # No, Strassen's algorithm uses 7 multiplications.
        # Each multiplication is of the form (linear combo of A) * (linear combo of B).
        # This is NOT a rank 1 term in the tensor decomposition of the multiplication map?
        # Wait. The tensor decomposition requires terms of the form u_i v_i w_i.
        # The product (a+b)(c+d) = ac + ad + bc + bd.
        # This requires 4 rank 1 terms if done directly.
        # But Strassen says we can do it with 7 products of sums.
        # How does this map to tensor rank?
        # The rank of the tensor is the number of terms in the decomposition.
        # If we compute M1 = (m00+m11)(p00+p11), this is one "multiplication" in the algorithm.
        # In the tensor decomposition, this corresponds to 4 terms if expanded, but we don't expand.
        # We keep it as a single term in the algorithmic sense?
        # NO. The tensor decomposition must be a sum of rank-1 tensors.
        # A term like (m00+m11)(p00+p11) w00 is NOT rank 1.
        # It is the sum of 4 rank 1 tensors.
        # So Strassen's algorithm does NOT directly give a tensor decomposition of rank 7.
        # It gives an algorithm with 7 multiplications.
        # The tensor rank of 2x2x2 is 7.
        # The decomposition is:
        # T = sum_{t=1}^7 u_t v_t w_t.
        # The u_t, v_t, w_t are vectors.
        # For Strassen, the vectors are:
        # u1 = (1,0,0,1), v1 = (1,0,0,1), w1 = (1,0,0,1) -> No, w is output.
        # Let's recall the explicit rank 7 decomposition of 2x2x2.
        
        # The 7 terms are:
        # 1. (A11+A22)(B11+B22) -> contributes to C11, C22
        # 2. (A21+A22)B11 -> contributes to C21, C22
        # 3. A11(B12-B22) -> contributes to C12, C22
        # 4. A22(B21-B11) -> contributes to C21, C11
        # 5. (A11+A12)B22 -> contributes to C12, C11
        # 6. (A21-A11)(B11+B12) -> contributes to C21, C22
        # 7. (A12-A22)(B21+B22) -> contributes to C11, C12
        
        # Each of these is a product of a linear form in A and a linear form in B.
        # Let L_A be the linear form in A, L_B be the linear form in B.
        # The term is L_A * L_B * (contribution to C).
        # This is a rank 1 tensor?
        # A rank 1 tensor is u v w.
        # Here, the term is (sum a_i u_i) (sum b_j v_j) (sum c_k w_k).
        # This is NOT a single rank 1 tensor u v w.
        # It is a sum of rank 1 tensors.
        # Specifically, (sum a_i u_i) (sum b_j v_j) = sum_{i,j} a_i b_j u_i v_j.
        # So each "Strassen multiplication" expands into multiple rank 1 terms.
        # Thus, the tensor rank is NOT 7.
        # The tensor rank of 2x2x2 is 7, but the algorithm uses 7 multiplications.
        # Wait. If the tensor rank is 7, there exists a decomposition with 7 terms.
        # The Strassen algorithm provides such a decomposition?
        # Yes, the 7 multiplications correspond to the 7 terms in the rank 7 decomposition.
        # But how?
        # The term M1 = (A11+A22)(B11+B22).
        # In the decomposition, this corresponds to the vector u1 = (1,0,0,1) for A?
        # No.
        # The rank 1 tensor u v w has entries u_i v_j w_k.
        # If we take u = (1, 0, 0, 1) (flattened A), v = (1, 0, 0, 1) (flattened B),
        # and w = (1, 0, 0, 1) (flattened C)?
        # Then the term is sum u_i v_j w_k A_i B_j?
        # No, the tensor T_{ijk} = delta_{ik} delta_{jk} for multiplication?
        # T = sum_{i,j,k} delta_{i,k} delta_{j,l} ...
        
        # Let's verify the rank 7 decomposition of 2x2x2.
        # The 7 terms are indeed rank 1 tensors.
        # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) is NOT correct because w must be a vector of length 4.
        # The contribution of M1 to C is:
        # C11 += M1, C22 += M1.
        # So w vector has 1 at C11 and 1 at C22.
        # u vector has 1 at A11 and 1 at A22.
        # v vector has 1 at B11 and 1 at B22.
        # Is (1,0,0,1) (1,0,0,1) (1,0,0,1) a rank 1 tensor?
        # Yes.
        # So M1 corresponds to ONE rank 1 term in the decomposition?
        # If so, the rank is 7.
        # And we just add these 7 terms to U, V, W.
        
        # Let's verify.
        # Term 1:
        # u: A11=1, A22=1. Others 0.
        # v: B11=1, B22=1. Others 0.
        # w: C11=1, C22=1. Others 0.
        # This term contributes 1*A11*1*B11*1*C11 + 1*A11*1*B22*1*C22 + ...
        # Wait. The product of u and v is a matrix.
        # u v^T = [[1,0,1,0], ...]
        # The tensor is u \otimes v \otimes w.
        # The contribution to C_{kl} is sum_{i,j} T_{ijk} A_i B_j.
        # For this term, T_{ijk} = u_i v_j w_k.
        # So C_{kl} gets contribution from A_i B_j if u_i v_j w_k != 0.
        # For C11 (k=0, l=0): w_0=1.
        # Contribution: sum_{i,j} u_i v_j A_i B_j.
        # u = (1,0,0,1) -> A11, A22.
        # v = (1,0,0,1) -> B11, B22.
        # Sum: A11B11 + A11B22 + A22B11 + A22B22.
        # This is (A11+A22)(B11+B22).
        # And it is added to C11.
        # Similarly for C22 (w_3=1).
        # So yes, each Strassen multiplication corresponds to exactly ONE rank 1 term in the tensor decomposition.
        
        # So we can simply add 7 terms for each 2x2x2 block.
        
        # Define the 7 Strassen terms (u, v, w coefficients)
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 10, 11
        # C indices: 00, 01, 10, 11
        
        # T1: (A00+A11)(B00+B11) -> C00, C11
        # u: [1,0,0,1], v: [1,0,0,1], w: [1,0,0,1]
        
        # T2: (A10+A11)B00 -> C10, C11
        # u: [0,0,1,1], v: [1,0,0,0], w: [0,0,1,1]
        
        # T3: A00(B01-B11) -> C01, C11
        # u: [1,0,0,0], v: [0,1,0,-1], w: [0,1,0,1]
        
        # T4: A11(B10-B00) -> C10, C00
        # u: [0,0,0,1], v: [0,0,1,-1], w: [1,0,1,0]
        
        # T5: (A00+A01)B11 -> C01, C00
        # u: [1,1,0,0], v: [0,0,0,1], w: [1,0,0,1]
        
        # T6: (A10-A00)(B00+B01) -> C10, C11
        # u: [-1,0,1,0], v: [1,1,0,0], w: [0,0,1,1]
        
        # T7: (A01-A11)(B10+B11) -> C00, C01
        # u: [0,1,0,-1], v: [0,0,1,1], w: [1,1,0,0]
        
        # Note: C indices in w are flattened: C00=0, C01=1, C10=2, C11=3.
        # But in our global W, the index is k*n + i.
        # C00 is i=0, k=0 -> idx 0.
        # C01 is i=0, k=1 -> idx 1.
        # C10 is i=1, k=0 -> idx 2.
        # C11 is i=1, k=1 -> idx 3.
        # This matches the local order if we map carefully.
        
        # Local A indices: 00, 01, 10, 11.
        # Global U indices: u00, u01, u10, u11.
        
        # Local B indices: 00, 01, 10, 11.
        # Global V indices: v00, v01, v10, v11.
        
        # Local C indices: 00, 01, 10, 11.
        # Global W indices: w00, w01, w10, w11.
        
        # We will add 7 terms.
        
        # Term 1
        U[u00][col] += 1; V[v00][col] += 1; W[w00][col] += 1
        U[u11][col] += 1; V[v11][col] += 1; W[w11][col] += 1
        col += 1
        
        # Term 2
        U[u10][col] += 1; V[v00][col] += 1; W[w10][col] += 1
        U[u11][col] += 1; V[v00][col] += 1; W[w11][col] += 1
        col += 1
        
        # Term 3
        U[u00][col] += 1; V[v01][col] += 1; W[w01][col] += 1
        U[u00][col] += 1; V[v11][col] += -1; W[w11][col] += 1
        col += 1
        
        # Term 4
        U[u11][col] += 1; V[v10][col] += 1; W[w10][col] += 1
        U[u11][col] += 1; V[v00][col] += -1; W[w00][col] += 1
        col += 1
        
        # Term 5
        U[u00][col] += 1; V[v11][col] += 1; W[w01][col] += 1
        U[u01][col] += 1; V[v11][col] += 1; W[w00][col] += 1
        col += 1
        
        # Term 6
        U[u10][col] += 1; V[v00][col] += 1; W[w10][col] += 1
        U[u00][col] += -1; V[v00][col] += 1; W[w10][col] += 1
        U[u10][col] += 1; V[v01][col] += 1; W[w11][col] += 1
        U[u00][col] += -1; V[v01][col] += 1; W[w11][col] += 1
        col += 1
        
        # Term 7
        U[u01][col] += 1; V[v10][col] += 1; W[w00][col] += 1
        U[u11][col] += -1; V[v10][col] += 1; W[w00][col] += 1
        U[u01][col] += 1; V[v11][col] += 1; W[w01][col] += 1
        U[u11][col] += -1; V[v11][col] += 1; W[w01][col] += 1
        col += 1

    # Apply to A_2L B_T
    # A_2L rows 1,2. Cols 0,1.
    # B_T rows 0,1. Cols 0,1 (Block 0), 2,3 (Block 1), 4,5 (Block 2).
    # C rows 1,2. Cols 0,1 (Block 0), 2,3 (Block 1), 4,5 (Block 2).
    
    a_rows = [1, 2]
    a_cols = [0, 1]
    b_rows = [0, 1]
    c_rows = [1, 2]
    
    for block in range(3):
        b_cols = [block*2, block*2+1]
        c_cols = [block*2, block*2+1]
        add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols)
        
    # Apply to A_2R B_B
    # A_2R rows 1,2. Cols 2,3.
    # B_B rows 2,3.
    # C rows 1,2.
    
    a_cols = [2, 3]
    b_rows = [2, 3]
    
    for block in range(3):
        b_cols = [block*2, block*2+1]
        c_cols = [block*2, block*2+1]
        add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
