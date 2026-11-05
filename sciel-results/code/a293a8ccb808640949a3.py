import numpy as np

def construct():
    n, m, p = 2, 4, 5
    
    # We will build the decomposition by splitting the middle dimension m=4 into two parts of size 2.
    # A is 2x4, B is 4x5.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x5.
    # C = A1*B1 + A2*B2.
    
    # We need a decomposition for 2x2 * 2x5.
    # We split B (2x5) into BL (2x2) and BR (2x3).
    # C_part = A*BL + A*BR.
    # A*BL uses Strassen (rank 7).
    # A*BR is 2x2 * 2x3. Split BR into BR1 (2x2) and BR2 (2x1).
    # A*BR1 uses Strassen (rank 7).
    # A*BR2 uses Trivial (rank 4).
    # Total rank for 2x2 * 2x5 is 7 + 7 + 4 = 18.
    
    # Total rank for 2x4 * 4x5 is 18 + 18 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: index in U (0..n*m-1)
    # v_idx: index in V (0..m*p-1)
    # w_idx: index in W (0..p*n-1)
    # coeff: coefficient
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff

    # We need to map the logical indices of the sub-matrices to the global indices.
    # Global A is 2x4. Indices: A[0,0..3], A[1,0..3].
    # A1 is A[:, 0:2], A2 is A[:, 2:4].
    # Global B is 4x5. Indices: B[0..3, 0..4].
    # B1 is B[0:2, :], B2 is B[2:4, :].
    # Global C is 2x5. Indices: C[0,0..4], C[1,0..4].
    # W index: k*n + i. k is 0..4 (col of C), i is 0..1 (row of C).
    
    # Strassen for 2x2 * 2x2
    # A_sub (2x2), B_sub (2x2)
    # A_sub indices relative to A: (r, c)
    # B_sub indices relative to B: (r, c)
    # C_sub indices relative to C: (r, c)
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        # A is at [a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        # B is at [b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        # C is at [c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        
        # M1 = (A00+A11)(B00+B11)
        add_term(a_row_offset*4 + a_col_offset, b_row_offset*5 + b_col_offset, (c_col_offset+0)*2 + (c_row_offset+0), 1, start_col)
        add_term(a_row_offset*4 + a_col_offset, b_row_offset*5 + b_col_offset+1, (c_col_offset+1)*2 + (c_row_offset+0), 1, start_col)
        add_term(a_row_offset*4 + a_col_offset+1, b_row_offset*5 + b_col_offset, (c_col_offset+0)*2 + (c_row_offset+1), 1, start_col)
        add_term(a_row_offset*4 + a_col_offset+1, b_row_offset*5 + b_col_offset+1, (c_col_offset+1)*2 + (c_row_offset+1), 1, start_col)
        
        # M2 = (A10+A11)B00
        add_term(a_row_offset*4 + a_col_offset+1, b_row_offset*5 + b_col_offset, (c_col_offset+0)*2 + (c_row_offset+1), 1, start_col+1)
        add_term(a_row_offset*4 + a_col_offset+1, b_row_offset*5 + b_col_offset, (c_col_offset+1)*2 + (c_row_offset+1), 1, start_col+1)
        add_term(a_row_offset*4 + a_col_offset, b_row_offset*5 + b_col_offset, (c_col_offset+0)*2 + (c_row_offset+1), 1, start_col+1)
        # Note: Strassen M2 is (A21+A22)B11 in 1-based. Here A10+A11 in 0-based row 1.
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # My add_term calls above for M1 were wrong. I need to sum the coefficients for the same (u,v,w) triple?
        # No, U, V, W are matrices.
        # For M1, we need (A00+A11) * (B00+B11).
        # This expands to A00B00 + A00B11 + A11B00 + A11B11.
        # This is 4 rank-1 terms.
        # But Strassen combines them into 1 multiplication.
        # In the tensor decomposition, each "multiplication" corresponds to a column in U, V, W.
        # The entries in U, V, W are the coefficients of the inputs.
        # So for M1, we have one column where:
        # U has 1 at A00 and A11.
        # V has 1 at B00 and B11.
        # W has 1 at C00 and C11? No, W distributes the result.
        # C00 gets M1. C11 gets M1.
        # So W has 1 at C00 and 1 at C11?
        # Let's check the convention.
        # C = sum_r U_r V_r W_r^T?
        # C_{ik} = sum_r U_{ik}^{(r)} V_{jk}^{(r)} ...
        # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}?
        # The trace convention W[k*n+i, r].
        # So C_{ik} = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r].
        
        # So for M1 = (A00+A11)(B00+B11) contributing to C00 and C11?
        # No, in Strassen, M1 contributes to C00, C11, C10, C01 with different signs.
        # M1 = (A00+A11)(B00+B11).
        # C00 += M1.
        # C11 += M1.
        # C10 -= M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for column r corresponding to M1:
        # U has 1 at A00, 1 at A11.
        # V has 1 at B00, 1 at B11.
        # W has 1 at C00, 1 at C11.
        
        # Let's re-implement add_strassen correctly.
        
        col = start_col
        # M1
        U[a_row_offset*4 + a_col_offset][col] += 1
        U[a_row_offset*4 + a_col_offset+1][col] += 1
        V[b_row_offset*5 + b_col_offset][col] += 1
        V[b_row_offset*5 + b_col_offset+1][col] += 1
        W[(c_col_offset+0)*2 + (c_row_offset+0)][col] += 1
        W[(c_col_offset+1)*2 + (c_row_offset+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        # A10 is (row 1, col 0). A11 is (row 1, col 1).
        U[a_row_offset*4 + a_col_offset+1][col] += 1 # A10? No, A[1,0]
        # Wait, A indices: row, col.
        # A00: (0,0). A01: (0,1). A10: (1,0). A11: (1,1).
        # M2 = (A10+A11)B00.
        U[a_row_offset*4 + a_col_offset+1][col] += 1 # A10
        U[a_row_offset*4 + a_col_offset+2][col] += 1 # A11? No, a_col_offset+1 is col 1.
        # a_col_offset is 0. A10 is (1,0). A11 is (1,1).
        # Index: 1*4 + 0 = 4. 1*4 + 1 = 5.
        U[a_row_offset*4 + a_col_offset+1][col] += 1 # This is A10 if a_col_offset=0?
        # a_row_offset=0. a_col_offset=0.
        # A10 is at 1*4+0 = 4.
        # A11 is at 1*4+1 = 5.
        # So:
        U[a_row_offset*4 + a_col_offset+1][col] = 0 # Reset? No, we are adding.
        
        # This manual indexing is error-prone.
        # I will use a helper function to get indices.
        
        return start_col + 7 # Placeholder

    # Given the complexity and risk of error in manual construction, 
    # and the time limit, I will stick to the trivial rank 40 
    # unless I can confidently implement a lower rank.
    # However, 36 is a significant improvement.
    # Let's try to implement the 36 rank carefully.
    
    pass
