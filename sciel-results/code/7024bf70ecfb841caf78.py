"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decompose M_{2,4,5} as A_L * B_T + A_R * B_B
    # A is 2x4: A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2)
    # B is 4x5: B_T = B[0:2, :] (2x5), B_B = B[2:4, :] (2x5)
    # C = A_L * B_T + A_R * B_B
    # A_L is 2x2, B_T is 2x5. We split B_T into two 2x2 blocks B_T1, B_T2 and a 2x1 block B_T3.
    # C_1 = A_L * B_T = A_L * B_T1 + A_L * B_T2 + A_L * B_T3
    # Rank of A_L * B_T1 (2x2 by 2x2) is 7 (Strassen).
    # Rank of A_L * B_T2 (2x2 by 2x2) is 7 (Strassen).
    # Rank of A_L * B_T3 (2x2 by 2x1) is 2 (Schoolbook).
    # So rank of A_L * B_T is 7+7+2 = 16.
    # Similarly, rank of A_R * B_B is 16.
    # Total rank = 32.
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Indexing conventions:
    # U has shape (n*m, rank) = (8, 32)
    # V has shape (m*p, rank) = (20, 32)
    # W has shape (p*n, rank) = (10, 32)
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen decomposition for 2x2x2 multiplication
    # Input indices:
    # A is 2x2. A[0,0]=a0, A[0,1]=a1, A[1,0]=a2, A[1,1]=a3
    # In U: row is i*m + j. Here m=2 for the sub-blocks.
    # But in the global U, the row is i*4 + j (since m=4 globally).
    # Let's map local indices to global indices.
    
    # For A_L (rows 0,1 of A; cols 0,1 of A):
    # Global row indices in U:
    # (i=0, j=0) -> 0*4+0 = 0
    # (i=0, j=1) -> 0*4+1 = 1
    # (i=1, j=0) -> 1*4+0 = 4
    # (i=1, j=1) -> 1*4+1 = 5
    
    # For A_R (rows 0,1 of A; cols 2,3 of A):
    # Global row indices in U:
    # (i=0, j=2) -> 0*4+2 = 2
    # (i=0, j=3) -> 0*4+3 = 3
    # (i=1, j=2) -> 1*4+2 = 6
    # (i=1, j=3) -> 1*4+3 = 7
    
    # For B_T (cols 0,1,2,3,4 of B; rows 0,1 of B):
    # Global col indices in V:
    # B_T1 (cols 0,1):
    #   (j=0, k=0) -> 0*5+0 = 0
    #   (j=0, k=1) -> 0*5+1 = 1
    #   (j=1, k=0) -> 1*5+0 = 5
    #   (j=1, k=1) -> 1*5+1 = 6
    # B_T2 (cols 2,3):
    #   (j=0, k=2) -> 0*5+2 = 2
    #   (j=0, k=3) -> 0*5+3 = 3
    #   (j=1, k=2) -> 1*5+2 = 7
    #   (j=1, k=3) -> 1*5+3 = 8
    # B_T3 (col 4):
    #   (j=0, k=4) -> 0*5+4 = 4
    #   (j=1, k=4) -> 1*5+4 = 9
    
    # For B_B (cols 0,1,2,3,4 of B; rows 2,3 of B):
    # Global col indices in V:
    # B_B1 (cols 0,1):
    #   (j=2, k=0) -> 2*5+0 = 10
    #   (j=2, k=1) -> 2*5+1 = 11
    #   (j=3, k=0) -> 3*5+0 = 15
    #   (j=3, k=1) -> 3*5+1 = 16
    # B_B2 (cols 2,3):
    #   (j=2, k=2) -> 2*5+2 = 12
    #   (j=2, k=3) -> 2*5+3 = 13
    #   (j=3, k=2) -> 3*5+2 = 17
    #   (j=3, k=3) -> 3*5+3 = 18
    # B_B3 (col 4):
    #   (j=2, k=4) -> 2*5+4 = 14
    #   (j=3, k=4) -> 3*5+4 = 19
    
    # Output C is 2x5.
    # Global row indices in W: W[k*n+i]
    # C[0,0] (i=0, k=0) -> 0*2+0 = 0
    # C[0,1] (i=0, k=1) -> 1*2+0 = 2
    # C[0,2] (i=0, k=2) -> 2*2+0 = 4
    # C[0,3] (i=0, k=3) -> 3*2+0 = 6
    # C[0,4] (i=0, k=4) -> 4*2+0 = 8
    # C[1,0] (i=1, k=0) -> 0*2+1 = 1
    # C[1,1] (i=1, k=1) -> 1*2+1 = 3
    # C[1,2] (i=1, k=2) -> 2*2+1 = 5
    # C[1,3] (i=1, k=3) -> 3*2+1 = 7
    # C[1,4] (i=1, k=4) -> 4*2+1 = 9
    
    def strassen_coeffs():
        # Returns list of (u_entries, v_entries, w_entries) for the 7 ranks
        # u_entries: dict mapping (i,j) -> coeff
        # v_entries: dict mapping (j,k) -> coeff
        # w_entries: dict mapping (i,k) -> coeff
        # Note: local indices for 2x2 matrix
        # u: A's elements, v: B's elements, w: C's elements
        
        # M1 = (a1+a0)(b0+b3)
        # M2 = (a2+a3)(b0)
        # M3 = (a1)(b3-b2)
        # M4 = (a3)(b2-b1)
        # M5 = (a3+a0)(b1+b2)
        # M6 = (a2-a1)(b1+b3)
        # M7 = (a1-a2)(b0+b1)
        
        # Let's define them as tuples of (row, col, coeff) for each matrix
        
        # M1
        u1 = [(0,0,1), (0,1,1)]
        v1 = [(0,0,1), (1,1,1)]
        w1 = [(0,0,1), (0,1,-1), (1,0,-1), (1,1,1)]
        
        # M2
        u2 = [(1,0,1), (1,1,1)]
        v2 = [(0,0,1)]
        w2 = [(0,0,1), (1,1,-1)]
        
        # M3
        u3 = [(0,1,1)]
        v3 = [(1,1,1), (1,0,-1)]
        w3 = [(0,0,1), (1,0,1)]
        
        # M4
        u4 = [(1,1,1)]
        v4 = [(1,0,1), (0,1,-1)]
        w4 = [(0,1,1), (1,1,1)]
        
        # M5
        u5 = [(0,0,1), (1,1,1)]
        v5 = [(0,1,1), (1,1,1)]
        w5 = [(0,1,-1), (1,0,1)]
        
        # M6
        u6 = [(1,0,1), (0,1,-1)]
        v6 = [(0,1,1), (1,1,1)]
        w6 = [(1,0,1), (1,1,-1)]
        
        # M7
        u7 = [(0,1,1), (1,0,-1)]
        v7 = [(0,0,1), (0,1,1)]
        w7 = [(0,1,-1), (1,0,-1)]
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
    
    strassen_decomps = strassen_coeffs()
    
    def schoolbook_2x2x1_coeffs():
        # A (2x2) * B (2x1)
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # Rank 2:
        # Rank 1:
        # u: (0,0,1), (1,0,1)  <-- Wait, standard schoolbook is 4 terms? No, 2 terms per output element.
        # Total rank is 2?
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # This is not rank 2 tensor decomposition in the bilinear sense unless we sum them?
        # No, the tensor is:
        # T = e_{0,0} \otimes f_{0,0} \otimes g_0 + e_{0,1} \otimes f_{0,1} \otimes g_0 + e_{1,0} \otimes f_{1,0} \otimes g_1 + e_{1,1} \otimes f_{1,1} \otimes g_1
        # This is 4 terms. Rank is 4?
        # Can we do it in fewer?
        # (A00*B0 + A01*B1) contributes to C0.
        # (A10*B0 + A11*B1) contributes to C1.
        # We need to represent 4 monomials.
        # Actually, for M_{2,2,1}, the rank is 2.
        # Why?
        # Let u1 = A00+A10, v1 = B0... no.
        # Let's check rank of M_{2,2,1}.
        # T_{i,j,k} = \delta_{k, 0} \delta_{i,j} + \dots? No.
        # T_{i,j,0} = \delta_{i, 0} \delta_{j, 0} + \dots ?
        # C_{i} = \sum_j A_{i,j} B_{j}
        # T_{i,j,k} = \delta_{k,0} \delta_{i,j}? No.
        # T_{i,j,0} = 1 if i==j? No.
        # T_{i,j,k} is 1 if i=j and k=0?
        # No, the tensor is T_{i,j,k} such that C_k = \sum_{i,j} A_i B_j T_{i,j,k}.
        # For k=0: C_0 = A_0 B_0. So T_{0,0,0}=1, T_{1,1,0}=1.
        # For k=1: C_1 = A_1 B_1. So T_{0,0,1}=0, T_{1,1,1}=1.
        # Wait, C is a vector.
        # C_0 = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}? No, B is 2x1.
        # B_0 is the top element, B_1 is the bottom element.
        # C_0 = A_{0,0} B_0 + A_{0,1} B_1
        # C_1 = A_{1,0} B_0 + A_{1,1} B_1
        # So T_{i,j,k} = \delta_{i, k} \dots? No.
        # T_{0,0,0} = 1, T_{0,1,0} = 1, T_{1,0,1} = 1, T_{1,1,1} = 1.
        # All other T are 0.
        # This tensor has 4 non-zero entries.
        # Is the rank 2?
        # Let's see if it can be sum of 2 rank 1 tensors.
        # Rank 1 tensor: u \otimes v \otimes w.
        # u_i v_j w_k.
        # We need to sum two such tensors to get the 4 entries.
        # Entry (0,0,0): 1. u1_0 v1_0 w1_0 + u2_0 v2_0 w2_0 = 1
        # Entry (0,1,0): 1. u1_0 v1_1 w1_0 + u2_0 v2_1 w2_0 = 1
        # Entry (1,0,1): 1. u1_1 v1_0 w1_1 + u2_1 v2_0 w2_1 = 1
        # Entry (1,1,1): 1. u1_1 v1_1 w1_1 + u2_1 v2_1 w2_1 = 1
        # Zeros elsewhere.
        # If we set w to (1,0) and (0,1)?
        # Then w1=(1,0), w2=(0,1).
        # Then terms with k=0 come only from term 1.
        # Term 1 must produce (0,0)=1 and (0,1)=1 for A,B.
        # So u1 \otimes v1 must be [[1, 1], [0, 0]].
        # u1 = [1, 0], v1 = [1, 1].
        # Check: u1_0 v1_0 = 1*1=1. u1_0 v1_1 = 1*1=1. u1_1 v1_0 = 0. u1_1 v1_1 = 0.
        # This works for C_0.
        # Now for C_1 (k=1), comes only from term 2.
        # Term 2 must produce (1,0)=1 and (1,1)=1 for A,B.
        # So u2 \otimes v2 must be [[0, 0], [1, 1]].
        # u2 = [0, 1], v2 = [0, 1].
        # Check: u2_1 v2_0 = 1*0=0? No, we need (1,0)=1.
        # (1,0) corresponds to i=1, j=0.
        # u2_1 v2_0 = 1 * v2_0. We need this to be 1. So v2_0 = 1.
        # (1,1) corresponds to i=1, j=1.
        # u2_1 v2_1 = 1 * v2_1. We need this to be 1. So v2_1 = 1.
        # So v2 = [1, 1].
        # Let's check zeros.
        # (0,0,1): u2_0 v2_0 w2_1 = 0 * 1 * 1 = 0. Correct.
        # (0,1,1): u2_0 v2_1 w2_1 = 0 * 1 * 1 = 0. Correct.
        # (1,0,0): u1_1 v1_0 w1_0 = 0 * 1 * 1 = 0. Correct.
        # (1,1,0): u1_1 v1_1 w1_0 = 0 * 1 * 1 = 0. Correct.
        # So yes, rank is 2.
        
        # Term 1:
        # u1: (0,0,1)
        # v1: (0,0,1), (0,1,1)
        # w1: (0,1) -> index 0
        # Term 2:
        # u2: (1,0,1)
        # v2: (1,0,1), (1,1,1)
        # w2: (1,1) -> index 1
        
        return [
            ([(0,0,1)], [(0,0,1), (0,1,1)], [(0,1)]),
            ([(1,0,1)], [(1,0,1), (1,1,1)], [(1,1)])
        ]
    
    schoolbook_decomps = schoolbook_2x2x1_coeffs()
    
    col_idx = 0
    
    # Process A_L * B_T
    # A_L is at rows 0,1; cols 0,1 of A.
    # B_T is at rows 0,1; cols 0,4 of B.
    
    # Add Strassen for A_L * B_T1
    for u, v, w in strassen_decomps:
        for r, c, coeff in u:
            # local i, j -> global i, j
            # A_L uses global j 0,1.
            # local i is same as global i.
            # local j 0 -> global j 0, local j 1 -> global j 1.
            global_r = r
            global_c = c
            U[global_r * m + global_c][col_idx] = coeff
        for r, c, coeff in v:
            # local j, k -> global j, k
            # B_T1 uses global j 0,1; k 0,1.
            # local j 0 -> global j 0, local j 1 -> global j 1.
            # local k 0 -> global k 0, local k 1 -> global k 1.
            global_r = r
            global_c = c
            V[global_r * p + global_c][col_idx] = coeff
        for r, c, coeff in w:
            # local i, k -> global i, k
            # C is 2x5.
            # local i 0,1 -> global i 0,1.
            # local k 0,1 -> global k 0,1.
            global_r = r
            global_c = c
            W[global_c * n + global_r][col_idx] = coeff
        col_idx += 1
        
    # Add Strassen for A_L * B_T2
    # B_T2 is at rows 0,1; cols 2,3 of B.
    for u, v, w in strassen_decomps:
        for r, c, coeff in u:
            # Same as before for U
            global_r = r
            global_c = c
            U[global_r * m + global_c][col_idx] = coeff
        for r, c, coeff in v:
            # local j, k -> global j, k
            # B_T2 uses global j 0,1; k 2,3.
            # local j 0 -> global j 0, local j 1 -> global j 1.
            # local k 0 -> global k 2, local k 1 -> global k 3.
            global_r = r
            global_c = c + 2
            V[global_r * p + global_c][col_idx] = coeff
        for r, c, coeff in w:
            # local i, k -> global i, k
            # C is 2x5.
            # local i 0,1 -> global i 0,1.
            # local k 0,1 -> global k 2,3.
            global_r = r
            global_c = c + 2
            W[global_c * n + global_r][col_idx] = coeff
        col_idx += 1
        
    # Add Schoolbook for A_L * B_T3
    # B_T3 is at rows 0,1; col 4 of B.
    for u, v, w in schoolbook_decomps:
        for r, c, coeff in u:
            global_r = r
            global_c = c
            U[global_r * m + global_c][col_idx] = coeff
        for r, c, coeff in v:
            # local j, k -> global j, k
            # B_T3 uses global j 0,1; k 4.
            # local j 0,1 -> global j 0,1.
            # local k 0 -> global k 4 (since k is always 4 in B_T3? No, B_T3 is 2x1.
            # The local k index in the 2x1 block is always 0.
            # Wait, the schoolbook decomposition I derived assumed B is 2x1 with indices 0,1.
            # So local k is always 0.
            # My coefficients for v were:
            # Term 1: (0,0,1), (0,1,1). Here c is the column index of B.
            # So for B_T3, global col is always 4.
            global_r = r
            global_c = 4
            V[global_r * p + global_c][col_idx] = coeff
        for r, c, coeff in w:
            # local i, k -> global i, k
            # local i 0,1 -> global i 0,1.
            # local k 0 -> global k 4.
            global_r = r
            global_c = 4
            W[global_c * n + global_r][col_idx] = coeff
        col_idx += 1
        
    # Process A_R * B_B
    # A_R is at rows 0,1; cols 2,3 of A.
    # B_B is at rows 2,3; cols 0,4 of B.
    
    # Add Strassen for A_R * B_B1
    for u, v, w in strassen_decomps:
        for r, c, coeff in u:
            # local i, j -> global i, j
            # A_R uses global i 0,1; global j 2,3.
            # local j 0 -> global j 2, local j 1 -> global j 3.
            global_r = r
            global_c = c + 2
            U[global_r * m + global_c][col_idx] = coeff
        for r, c, coeff in v:
            # local j, k -> global j, k
            # B_B1 uses global j 2,3; global k 0,1.
            # local j 0 -> global j 2, local j 1 -> global j 3.
            # local k 0 -> global k 0, local k 1 -> global k 1.
            global_r = r + 2
            global_c = c
            V[global_r * p + global_c][col_idx] = coeff
        for r, c, coeff in w:
            # local i, k -> global i, k
            global_r = r
            global_c = c
            W[global_c * n + global_r][col_idx] = coeff
        col_idx += 1
        
    # Add Strassen for A_R * B_B2
    for u, v, w in strassen_decomps:
        for r, c, coeff in u:
            global_r = r
            global_c = c + 2
            U[global_r * m + global_c][col_idx] = coeff
        for r, c, coeff in v:
            # B_B2 uses global j 2,3; global k 2,3.
            global_r = r + 2
            global_c = c + 2
            V[global_r * p + global_c][col_idx] = coeff
        for r, c, coeff in w:
            # C uses global i 0,1; global k 2,3.
            global_r = r
            global_c = c + 2
            W[global_c * n + global_r][col_idx] = coeff
        col_idx += 1
        
    # Add Schoolbook for A_R * B_B3
    for u, v, w in schoolbook_decomps:
        for r, c, coeff in u:
            global_r = r
            global_c = c + 2
            U[global_r * m + global_c][col_idx] = coeff
        for r, c, coeff in v:
            # B_B3 uses global j 2,3; global k 4.
            global_r = r + 2
            global_c = 4
            V[global_r * p + global_c][col_idx] = coeff
        for r, c, coeff in w:
            # C uses global i 0,1; global k 4.
            global_r = r
            global_c = 4
            W[global_c * n + global_r][col_idx] = coeff
        col_idx += 1
        
    # Verify col_idx is 32
    # 7 (A_L*B_T1) + 7 (A_L*B_T2) + 2 (A_L*B_T3) = 16
    # 7 (A_R*B_B1) + 7 (A_R*B_B2) + 2 (A_R*B_B3) = 16
    # Total 32.
    
    if col_idx != 32:
        raise ValueError(f"Expected rank 32, got {col_idx}")

    # Convert to integer if possible
    for mat in (U, V, W):
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == int(mat[i][j]):
                    mat[i][j] = int(mat[i][j])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
