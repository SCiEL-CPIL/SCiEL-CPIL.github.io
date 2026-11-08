import numpy as np

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the 2-level Strassen algorithm (Kronecker product of 2x2 Strassen factors).
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # 2x2 Strassen factors (rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # W2 indexed as W2[col*2+row, k]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    U2[0, 0] = 1; U2[3, 0] = 1  # A00 + A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00 + B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11 (col*2+row: 0*2+0=0, 1*2+1=3)
    
    # Term 1: M2
    U2[2, 1] = 1; U2[3, 1] = 1  # A10 + A11
    V2[0, 1] = 1  # B00
    W2[2, 1] = 1  # C10 (col*2+row: 0*2+1=2)
    W2[3, 1] = -1  # C11
    
    # Term 2: M3
    U2[0, 2] = 1  # A00
    V2[1, 2] = 1; V2[3, 2] = -1  # B01 - B11
    W2[1, 2] = 1  # C01 (col*2+row: 1*2+0=1)
    W2[3, 2] = 1  # C11
    
    # Term 3: M4
    U2[3, 3] = 1  # A11
    V2[2, 3] = 1; V2[0, 3] = -1  # B10 - B00
    W2[0, 3] = 1  # C00
    W2[2, 3] = 1  # C10
    
    # Term 4: M5
    U2[0, 4] = 1; U2[1, 4] = 1  # A00 + A01
    V2[3, 4] = 1  # B11
    W2[0, 4] = -1  # C00
    W2[1, 4] = 1  # C01
    
    # Term 5: M6
    U2[2, 5] = 1; U2[0, 5] = -1  # A10 - A00
    V2[0, 5] = 1; V2[1, 5] = 1  # B00 + B01
    W2[3, 5] = 1  # C11
    
    # Term 6: M7
    U2[1, 6] = 1; U2[3, 6] = -1  # A01 - A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10 + B11
    W2[0, 6] = 1  # C00
    
    # Now use Kronecker product to get 4x4 factors
    # U = kron(U2, U2), V = kron(V2, V2), W = kron(W2, W2)
    # But we need to be careful with the indexing convention.
    
    # A is 4x4, indexed as A[r,c] where r,c in 0..3
    # U has shape (16, 49), indexed as U[r*4+c, k]
    
    # B is 4x4, indexed as B[r,c] where r,c in 0..3
    # V has shape (16, 49), indexed as V[r*4+c, k]
    
    # C is 4x4, indexed as C[r,c] where r,c in 0..3
    # W has shape (16, 49), indexed as W[c*4+r, k] (trace convention: W[k*n+i, r] where output is C[r,c])
    # Wait, the task says W[k*n+i,r] where the output factor follows the official trace convention.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # This means W[i, k] where i = k*n + r... no wait.
    # W has shape (p*n, rank) = (16, 49)
    # The convention is W[k*n+i, r]... hmm, this is confusing.
    
    # Let me think about this more carefully.
    # The tensor T[i,j,k] for matrix multiplication is T[i,j,k] = 1 if i*k + j = i' where...
    # Actually, for C = A*B, we have C[i,j] = sum_k A[i,k]*B[k,j]
    # So the tensor is T[i,j,k] where the three indices are (i for A row, j for B col, k for contraction)
    # But the factorization is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector in R^(n*m) for A, v_r in R^(m*p) for B, w_r in R^(p*n) for C.
    
    # The indexing convention:
    # A is flattened as A[r*m+c] = A[r,c]
    # B is flattened as B[r*p+c] = B[r,c]
    # C is flattened as... the task says W[k*n+i, r]
    
    # I think the convention is:
    # C[i,j] is stored at index j*n + i in the W vector (column-major for C)
    # So W[j*4+i, k] is the coefficient for C[i,j] in term k.
    
    # Let me verify: if C = A*B, then C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor T has T[i, j, k] = delta_{i,k} * delta_{k,j}... no.
    # T[A_flat, B_flat, C_flat] where A_flat = i*m+k, B_flat = k*p+j, C_flat = ?
    
    # For the decomposition to work:
    # sum_k U[A_flat, k] * V[B_flat, k] * W[C_flat, k] = T[A_flat, B_flat, C_flat]
    
    # T[A_flat, B_flat, C_flat] = 1 if A_flat = i*m+k, B_flat = k'*p+j, C_flat = ? and i=k', k=j... 
    # Actually T = 1 if the indices are consistent: A[i,k], B[k,j], C[i,j]
    # So T[i*m+k, k*p+j, C_index] = 1 where C_index corresponds to C[i,j]
    
    # If C is stored column-major: C_index = j*n + i
    # Then T[i*m+k, k*p+j, j*n+i] = 1
    
    # Let me just use the Kronecker product approach and verify.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # For the 2-level Strassen:
    # We split A into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # The Kronecker product approach:
    # U4 = kron(U2, U2) but with appropriate reshaping
    
    # Actually, let me think about this differently.
    # The 4x4 multiplication can be seen as a 2x2 block multiplication where each block is 2x2.
    # The Strassen algorithm at the block level gives 7 block products.
    # Each block product is a 2x2x2 multiplication, which itself needs rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The U factor: for each of the 7 block-level terms, we have a 2x2 linear form on the blocks.
    # Each block is 2x2, so the linear form on a block is a 4-dimensional vector.
    # The total U vector is 16-dimensional (4 blocks * 4 entries each).
    
    # Let me construct this properly.
    
    # Block indexing:
    # Block (br, bc) of A has entries A[2*br+r, 2*bc+c] for r,c in 0,1
    # Flat index: (2*br+r)*4 + (2*bc+c)
    
    # For Strassen at the block level:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block product Mk, we need to compute the 2x2 matrix product using Strassen (rank 7).
    # So for each block-level term, we have 7 sub-terms.
    
    # Total terms: 7 * 7 = 49
    
    # Let me define the block-level Strassen coefficients:
    # For block (br, bc), the 2x2 matrix has entries at:
    # (2*br, 2*bc), (2*br, 2*bc+1), (2*br+1, 2*bc), (2*br+1, 2*bc+1)
    # Flat indices: (2*br)*4 + (2*bc), (2*br)*4 + (2*bc+1), (2*br+1)*4 + (2*bc), (2*br+1)*4 + (2*bc+1)
    
    def block_indices(br, bc):
        """Return flat indices for the 2x2 block (br, bc)"""
        return [
            (2*br)*4 + (2*bc),
            (2*br)*4 + (2*bc+1),
            (2*br+1)*4 + (2*bc),
            (2*br+1)*4 + (2*bc+1)
        ]
    
    # Strassen 2x2 sub-algorithm indices (local to a 2x2 block):
    # s0: (X00+X11)(Y00+Y11) -> local indices 0,3 for X; 0,3 for Y
    # s1: (X10+X11)(Y00) -> local indices 2,3 for X; 0 for Y
    # s2: (X00)(Y01-Y11) -> local index 0 for X; 1,3 for Y
    # s3: (X11)(Y10-Y00) -> local index 3 for X; 2,0 for Y
    # s4: (X00+X01)(Y11) -> local indices 0,1 for X; 3 for Y
    # s5: (X10-X00)(Y00+Y01) -> local indices 2,0 for X; 0,1 for Y
    # s6: (X01-X11)(Y10+Y11) -> local indices 1,3 for X; 2,3 for Y
    
    # C contributions for 2x2 Strassen:
    # Z00 = s0 + s3 - s4 + s6
    # Z01 = s2 + s4
    # Z10 = s1 + s3
    # Z11 = s0 - s1 + s2 + s5
    
    # Local C indices: 0->Z00, 1->Z01, 2->Z10, 3->Z11
    # But Z is stored column-major: Z[0]=Z00, Z[1]=Z10, Z[2]=Z01, Z[3]=Z11
    # Wait, for a 2x2 matrix Z, column-major storage:
    # Z[0] = Z[0,0], Z[1] = Z[1,0], Z[2] = Z[0,1], Z[3] = Z[1,1]
    # So Z00 -> index 0, Z10 -> index 1, Z01 -> index 2, Z11 -> index 3
    
    # Let me define the Strassen coefficients more carefully.
    # For a 2x2 matrix X with entries X00, X01, X10, X11 (row-major):
    # Flat index: X[0]=X00, X[1]=X01, X[2]=X10, X[3]=X11
    
    # Strassen terms:
    # s0: (X00+X11)(Y00+Y11)
    # s1: (X10+X11)(Y00)
    # s2: (X00)(Y01-Y11)
    # s3: (X11)(Y10-Y00)
    # s4: (X00+X01)(Y11)
    # s5: (X10-X00)(Y00+Y01)
    # s6: (X01-X11)(Y10+Y11)
    
    # Z = X*Y
    # Z00 = X00*Y00 + X01*Y10
    # Z01 = X00*Y01 + X01*Y11
    # Z10 = X10*Y00 + X11*Y10
    # Z11 = X10*Y01 + X11*Y11
    
    # Check:
    # s0 + s3 - s4 + s6 = (X00+X11)(Y00+Y11) + X11(Y10-Y00) - (X00+X01)Y11 + (X01-X11)(Y10+Y11)
    # = X00Y00 + X00Y11 + X11Y00 + X11Y11 + X11Y10 - X11Y00 - X00Y11 - X01Y11 + X01Y10 + X01Y11 - X11Y10 - X11Y11
    # = X00Y00 + X01Y10 ✓
    
    # s2 + s4 = X00(Y01-Y11) + (X00+X01)Y11 = X00Y01 - X00Y11 + X00Y11 + X01Y11 = X00Y01 + X01Y11 ✓
    
    # s1 + s3 = (X10+X11)Y00 + X11(Y10-Y00) = X10Y00 + X11Y00 + X11Y10 - X11Y00 = X10Y00 + X11Y10 ✓
    
    # s0 - s1 + s2 + s5 = (X00+X11)(Y00+Y11) - (X10+X11)Y00 + X00(Y01-Y11) + (X10-X00)(Y00+Y01)
    # = X00Y00 + X00Y11 + X11Y00 + X11Y11 - X10Y00 - X11Y00 + X00Y01 - X00Y11 + X10Y00 + X10Y01 - X00Y00 - X00Y01
    # = X11Y11 + X10Y01 ✓
    
    # Great, the Strassen algorithm is correct.
    
    # Now for the 4x4 case:
    # We have 7 block-level terms, each contributing to one or more 2x2 blocks of C.
    # For each block-level term, we compute a 2x2 matrix product using Strassen (7 sub-terms).
    
    # Let me define the block-level Strassen:
    # For block (br, bc) of A and block (br', bc') of B:
    # The product contributes to block (br, bc') of C.
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)(B00) -> contributes to C10, C11
    # M3 = (A00)(B01-B11) -> contributes to C01, C11
    # M4 = (A11)(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)(B11) -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # Wait, I need to be more careful.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00:
    # M1 + M4 - M5 + M7 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A01B10 ✓
    
    # C01:
    # M3 + M5 = A00(B01-B11) + (A00+A01)B11 = A00B01 - A00B11 + A00B11 + A01B11 = A00B01 + A01B11 ✓
    
    # C10:
    # M2 + M4 = (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10 ✓
    
    # C11:
    # M1 - M2 + M3 + M6 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A11B11 + A10B01 ✓
    
    # Now, each Mk is a product of two 2x2 matrices (linear combinations of blocks).
    # To compute Mk, we use the 2x2 Strassen algorithm (7 sub-terms).
    
    # For the U factor:
    # For each block-level term Mk, the A-side is a linear combination of 2x2 blocks.
    # Each block is 2x2, so the A-side vector has 16 entries (4 blocks * 4 entries).
    # For each of the 7 sub-terms of the 2x2 Strassen, we have a specific linear form.
    
    # Let me define this systematically.
    
    # Block-level Strassen A-side coefficients for each Mk:
    # Mk: (block_br, block_bc) for A, with coefficients
    # M1: A00 + A11 -> blocks (0,0) and (1,1) with coeff 1
    # M2: A10 + A11 -> blocks (1,0) and (1,1) with coeff 1
    # M3: A00 -> block (0,0) with coeff 1
    # M4: A11 -> block (1,1) with coeff 1
    # M5: A00 + A01 -> blocks (0,0) and (0,1) with coeff 1
    # M6: A10 - A00 -> blocks (1,0) and (0,0) with coeffs 1, -1
    # M7: A01 - A11 -> blocks (0,1) and (1,1) with coeffs 1, -1
    
    # For each Mk, the 2x2 Strassen sub-terms have specific local coefficients.
    # Local index 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    
    # Strassen sub-term A-side local coefficients:
    # s0: [1, 0, 0, 1]  (X00 + X11)
    # s1: [0, 0, 1, 1]  (X10 + X11)
    # s2: [1, 0, 0, 0]  (X00)
    # s3: [0, 0, 0, 1]  (X11)
    # s4: [1, 1, 0, 0]  (X00 + X01)
    # s5: [-1, 0, 1, 0] (X10 - X00)
    # s6: [0, 1, 0, -1] (X01 - X11)
    
    # Strassen sub-term B-side local coefficients:
    # s0: [1, 0, 0, 1]  (Y00 + Y11)
    # s1: [1, 0, 0, 0]  (Y00)
    # s2: [0, 1, 0, -1] (Y01 - Y11)
    # s3: [0, 0, 1, -1] wait, Y10 - Y00 -> local indices: Y00=0, Y01=1, Y10=2, Y11=3
    # So Y10 - Y00 = [0, 0, 1, -1]? No: Y00 is index 0, Y10 is index 2.
    # So Y10 - Y00 has coeff -1 at index 0, +1 at index 2: [-1, 0, 1, 0]
    # s4: [0, 0, 0, 1]  (Y11)
    # s5: [1, 1, 0, 0]  (Y00 + Y01)
    # s6: [0, 0, 1, 1]  (Y10 + Y11)
    
    # Let me re-verify s3: X11 * (Y10 - Y00)
    # Y10 is local index 2, Y00 is local index 0
    # So B-side: [-1, 0, 1, 0]
    
    # Strassen sub-term C-side local coefficients (column-major: 0=Z00, 1=Z10, 2=Z01, 3=Z11):
    # s0: [1, 0, 0, 1]  (Z00 + Z11)
    # s1: [0, 1, 0, -1] (Z10 - Z11)
    # s2: [0, 0, 1, 1]  (Z01 + Z11)
    # s3: [1, 1, 0, 0]  (Z00 + Z10)
    # s4: [-1, 0, 1, 0] (Z01 - Z00)
    # s5: [0, 0, 0, 1]  (Z11)
    # s6: [1, 0, 0, 0]  (Z00)
    
    # Now let me build the full 4x4 factors.
    
    # For each block-level term Mk (k = 0..6) and each sub-term s (s = 0..6):
    # Total term index: k * 7 + s
    
    # A-side:
    # For block (br, bc) with coefficient c in Mk:
    #   For local index li with coefficient cl in sub-term s:
    #     Flat index = (2*br + li//2)*4 + (2*bc + li%2)
    #     U[flat_index, k*7+s] += c * cl
    
    # B-side: similar
    
    # C-side:
    # For block (br, bc) of C with coefficient c in Mk:
    #   For local index li with coefficient cl in sub-term s:
    #     The C entry is C[2*br + li//2, 2*bc + li%2]
    #     Column-major index = (2*bc + li%2)*4 + (2*br + li//2)
    #     W[col_major_index, k*7+s] += c * cl
    
    # Block-level Strassen definitions:
    # Mk: A_blocks, B_blocks, C_blocks
    # Each is a list of (br, bc, coeff)
    
    block_strassen = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)(B00) -> C10, C11
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)]
        },
        # M3: (A00)(B01-B11) -> C01, C11
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(0,1,1), (1,1,1)]
        },
        # M4: (A11)(B10-B00) -> C00, C10
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)]
        },
        # M5: (A00+A01)(B11) -> C00, C01
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)]
        },
        # M6: (A10-A00)(B00+B01) -> C11
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11) -> C00
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1)]
        },
    ]
    
    # Sub-term coefficients
    A_sub = [
        [1, 0, 0, 1],  # s0
        [0, 0, 1, 1],  # s1
        [1, 0, 0, 0],  # s2
        [0, 0, 0, 1],  # s3
        [1, 1, 0, 0],  # s4
        [-1, 0, 1, 0], # s5
        [0, 1, 0, -1], # s6
    ]
    
    B_sub = [
        [1, 0, 0, 1],  # s0
        [1, 0, 0, 0],  # s1
        [0, 1, 0, -1], # s2
        [-1, 0, 1, 0], # s3
        [0, 0, 0, 1],  # s4
        [1, 1, 0, 0],  # s5
        [0, 0, 1, 1],  # s6
    ]
    
    C_sub = [
        [1, 0, 0, 1],  # s0
        [0, 1, 0, -1], # s1
        [0, 0, 1, 1],  # s2
        [1, 1, 0, 0],  # s3
        [-1, 0, 1, 0], # s4
        [0, 0, 0, 1],  # s5
        [1, 0, 0, 0],  # s6
    ]
    
    # Build the factors
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for k, block_term in enumerate(block_strassen):
        for s in range(7):
            term_idx = k * 7 + s
            
            # A-side
            for (br, bc, c_coeff) in block_term['A']:
                for li in range(4):
                    a_coeff = A_sub[s][li]
                    if a_coeff != 0:
                        row = 2*br + li//2
                        col = 2*bc + li%2
                        flat = row*4 + col
                        U[flat, term_idx] += c_coeff * a_coeff
            
            # B-side
            for (br, bc, c_coeff) in block_term['B']:
                for li in range(4):
                    b_coeff = B_sub[s][li]
                    if b_coeff != 0:
                        row = 2*br + li//2
                        col = 2*bc + li%2
                        flat = row*4 + col
                        V[flat, term_idx] += c_coeff * b_coeff
            
            # C-side
            for (br, bc, c_coeff) in block_term['C']:
                for li in range(4):
                    c_coeff_sub = C_sub[s][li]
                    if c_coeff_sub != 0:
                        row = 2*br + li//2
                        col = 2*bc + li%2
                        # Column-major: col*4 + row
                        flat = col*4 + row
                        W[flat, term_idx] += c_coeff * c_coeff_sub
    
    return {"U": U, "V": V, "W": W}
