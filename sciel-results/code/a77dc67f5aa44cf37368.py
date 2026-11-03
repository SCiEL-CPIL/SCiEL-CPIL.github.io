import numpy as np

def construct():
    n, m, p = 2, 3, 8
    # Rank 44 construction
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a bilinear term
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # Term: A[i,j] * B[j,k] contributes to C[i,k]
    # In tensor: U[idx_u, col] = 1, V[idx_v, col] = 1, W[idx_w, col] = 1
    # But Strassen involves sums and differences.
    # We will build the terms directly.
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Output C11 = M1 + M4 - M5 + M7
    # Output C12 = M3 + M5
    # Output C21 = M2 + M4
    # Output C22 = M1 - M2 + M3 + M6
    
    # We define the 7 multiplications for a 2x2x2 block.
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    def add_strassen_block(a_base, b_base, c_base):
        nonlocal col
        # a_base: index of A[0,0] in global A (2x3) -> row i, col j
        # b_base: index of B[0,0] in global B (3x8) -> row j, col k
        # c_base: index of C[0,0] in global C (2x8) -> row i, col k
        
        # A is 2x3. a_base is (i, j). A[i, j+1] is a_base + 1?
        # A indices: i in 0..1, j in 0..2.
        # If we take A_L (2x2), it uses j=0,1.
        # a_base corresponds to A[i0, j0].
        # A00 = A[i0, j0], A01 = A[i0, j0+1], A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
        
        # B is 3x8. b_base is (j0, k0).
        # B_top (2x4) uses j=0,1.
        # B00 = B[j0, k0], B01 = B[j0, k0+1], B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
        
        # C is 2x8. c_base is (i0, k0).
        # C00 = C[i0, k0], C01 = C[i0, k0+1], C10 = C[i0+1, k0], C11 = C[i0+1, k0+1]
        
        # M1: (A00+A11)(B00+B11)
        U[a_base, col] += 1
        U[a_base + 1 + 1, col] += 1 # A11 is i0+1, j0+1 -> (i0+1)*3 + (j0+1)
        # Wait, a_base is i0*m + j0.
        # A11 index: (i0+1)*m + (j0+1)
        U[(a_base // m + 1) * m + (a_base % m + 1), col] += 1
        
        V[b_base, col] += 1
        V[(b_base // p + 1) * p + (b_base % p + 1), col] += 1 # B11
        
        W[c_base, col] += 1
        W[(c_base // n + 1) * n + (c_base % n + 1), col] += 1 # C11
        W[c_base + 1, col] += 1 # C01? No.
        # C11 is (i0+1, k0+1).
        # C00 is (i0, k0).
        # C01 is (i0, k0+1).
        # C10 is (i0+1, k0).
        
        # Let's use explicit index calculation to avoid errors.
        i0, j0 = divmod(a_base, m)
        j0_b, k0 = divmod(b_base, p) # b_base is j*p+k
        i0_c, k0_c = divmod(c_base, n) # c_base is k*n+i ? 
        # W index is k*n + i.
        # So c_base = k0*n + i0.
        
        # M1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0_b*p + k0, col] += 1
        V[(j0_b+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1 # C00
        W[(k0+1)*n + (i0+1), col] += 1 # C11
        col += 1
        
        # M2: (A10+A11)B00
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0_b*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1 # C10
        col += 1
        
        # M3: A00(B01-B11)
        U[i0*m + j0, col] += 1
        V[j0_b*p + (k0+1), col] += 1
        V[(j0_b+1)*p + (k0+1), col] -= 1
        W[k0*n + i0, col] += 1 # C00
        W[(k0+1)*n + i0, col] += 1 # C01
        col += 1
        
        # M4: A11(B10-B00)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0_b+1)*p + k0, col] += 1
        V[j0_b*p + k0, col] -= 1
        W[k0*n + (i0+1), col] += 1 # C10
        W[(k0+1)*n + (i0+1), col] += 1 # C11
        col += 1
        
        # M5: (A00+A01)B11
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0_b+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1 # C00
        W[(k0+1)*n + i0, col] += 1 # C01
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0_b*p + k0, col] += 1
        V[j0_b*p + (k0+1), col] += 1
        W[k0*n + (i0+1), col] += 1 # C10
        W[(k0+1)*n + (i0+1), col] += 1 # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0_b+1)*p + k0, col] += 1
        V[(j0_b+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1 # C00
        W[(k0+1)*n + i0, col] += 1 # C01
        col += 1
        
        # Now we have added 7 terms to col.
        # But we need to combine them into C.
        # The above code added contributions to W for each M.
        # M1 contributed to C00 and C11.
        # M2 contributed to C10.
        # M3 contributed to C00 and C01.
        # M4 contributed to C10 and C11.
        # M5 contributed to C00 and C01.
        # M6 contributed to C10 and C11.
        # M7 contributed to C00 and C01.
        
        # This is incorrect. Strassen combines the M's to form C.
        # The tensor decomposition requires that the sum of the terms equals C.
        # My `add_strassen_block` logic above is flawed because it treats each M as a separate rank-1 term that contributes to multiple C entries, which is not how the tensor rank decomposition works directly in this additive way without careful coefficient tracking.
        
        # Correct approach:
        # The rank-1 terms are the M's.
        # M1 = (A00+A11)(B00+B11). This is one multiplication.
        # It contributes to C00, C01, C10, C11?
        # No, M1 is a scalar.
        # C00 = M1 + ...
        # So the term for M1 in the tensor is:
        # U vector has 1 at A00, 1 at A11.
        # V vector has 1 at B00, 1 at B11.
        # W vector has 1 at C00, 1 at C11? No.
        # The bilinear form is <U, A> <V, B> W.
        # If W has 1 at C00, then M1 adds to C00.
        # But M1 also adds to C11?
        # In Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C00 = M1 - M2 + M3 + M6 (Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Let's check C00 (which is C11 in 1-based? No, C11 is top-left).
        # Standard:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        #
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So, for each Pk, we have a rank-1 term.
        # The W vector for Pk has non-zero entries at the C positions where Pk appears.
        # P1 appears in C11 (coeff 1) and C22 (coeff 1).
        # So W for P1 has 1 at C11 and 1 at C22.
        
        # Let's re-implement add_strassen_block correctly.
        
        col -= 7 # Reset col
        i0, j0 = divmod(a_base, m)
        j0_b, k0 = divmod(b_base, p)
        i0_c, k0_c = divmod(c_base, n)
        
        # P1: (A00+A11)(B00+B11) -> C11, C22
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0_b*p + k0, col] += 1
        V[(j0_b+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1 # C11 (i0, k0)
        W[(k0+1)*n + (i0+1), col] += 1 # C22 (i0+1, k0+1)
        col += 1
        
        # P2: (A10+A11)B00 -> C21
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0_b*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1 # C21 (i0+1, k0)
        col += 1
        
        # P3: A00(B01-B11) -> C12
        U[i0*m + j0, col] += 1
        V[j0_b*p + (k0+1), col] += 1
        V[(j0_b+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1 # C12 (i0, k0+1)
        col += 1
        
        # P4: A11(B10-B00) -> C11, C21
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0_b+1)*p + k0, col] += 1
        V[j0_b*p + k0, col] -= 1
        W[k0*n + i0, col] += 1 # C11
        W[k0*n + (i0+1), col] += 1 # C21
        col += 1
        
        # P5: (A00+A01)B11 -> C11, C12
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0_b+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1 # C11
        W[(k0+1)*n + i0, col] += 1 # C12
        col += 1
        
        # P6: (A10-A00)(B00+B01) -> C22
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0_b*p + k0, col] += 1
        V[j0_b*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1 # C22
        col += 1
        
        # P7: (A01-A11)(B10+B11) -> C11, C12
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0_b+1)*p + k0, col] += 1
        V[(j0_b+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1 # C11
        W[(k0+1)*n + i0, col] += 1 # C12
        col += 1

    # Now we call this for the blocks.
    # Block 1: A_L (i=0,1; j=0,1) * B_1_top (j=0,1; k=0,1,2,3)
    # B_1_top is 2x4. Split into B_11 (2x2, k=0,1) and B_12 (2x2, k=2,3).
    # C_1 is 2x4. C_11 (k=0,1) and C_12 (k=2,3).
    
    # For B_11: a_base = 0 (i=0, j=0), b_base = 0 (j=0, k=0), c_base = 0 (k=0, i=0)
    add_strassen_block(0, 0, 0)
    
    # For B_12: a_base = 0, b_base = 2 (j=0, k=2), c_base = 2 (k=2, i=0)
    add_strassen_block(0, 2, 2)
    
    # A_R B_1_bottom: A (i=0,1; j=2) * B_1_bottom (j=2; k=0,1,2,3)
    # 8 terms.
    for k in range(4):
        # C[0, k] += A[0, 2] B[2, k]
        U[0*m + 2, col] += 1
        V[2*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        # C[1, k] += A[1, 2] B[2, k]
        U[1*m + 2, col] += 1
        V[2*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1

    # Block 2: A_L * B_2_top (k=4,5,6,7)
    # B_21 (k=4,5), B_22 (k=6,7)
    # C_21 (k=4,5), C_22 (k=6,7)
    
    # For B_21: a_base = 0, b_base = 4 (j=0, k=4), c_base = 4 (k=4, i=0)
    add_strassen_block(0, 4, 4)
    
    # For B_22: a_base = 0, b_base = 6 (j=0, k=6), c_base = 6 (k=6, i=0)
    add_strassen_block(0, 6, 6)
    
    # A_R B_2_bottom: k=4,5,6,7
    for k in range(4, 8):
        U[0*m + 2, col] += 1
        V[2*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        U[1*m + 2, col] += 1
        V[2*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
