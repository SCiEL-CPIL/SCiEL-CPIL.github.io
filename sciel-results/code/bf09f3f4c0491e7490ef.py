"""
Matrix multiplication 4x4 by 4x7: find a low-rank exact tensor decomposition.
Uses Strassen's algorithm for the 2x2 block structure.
"""
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4 (rows of A), m=4 (cols of A / rows of B), p=7 (cols of B)
    
    # We'll decompose the multiplication A(n x m) * B(m x p) = C(n x p)
    # into a sum of rank-1 tensors.
    # 
    # Strategy: Use block decomposition.
    # A is 4x4, B is 4x7.
    # Split A into 2x2 blocks of 2x2 submatrices.
    # Split B into 2x2 blocks where we handle the 7 columns specially.
    #
    # Let's use the following approach:
    # Split rows of A/C into two groups of 2.
    # Split cols of B into groups: first 4 cols as 2x2 blocks, last 3 cols handled separately.
    #
    # Actually, let's think more carefully. We want to minimize rank.
    #
    # One clean approach: Use Strassen on 2x2 blocks.
    # For a 2x2 * 2x2 multiplication, Strassen uses 7 multiplications.
    #
    # Let's partition:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x? 
    #
    # The issue is B is 4x7. Let's split B's columns into two groups: first 4 and last 3.
    # B_left = first 4 cols (4x4), B_right = last 3 cols (4x3).
    #
    # C = A * B = A * B_left + A * B_right
    #
    # For A * B_left: both are 4x4. We can use Strassen's algorithm.
    # For A * B_right: A is 4x4, B_right is 4x3. 
    #
    # Let's handle A * B_left (4x4 * 4x4) first with Strassen.
    # Then handle A * B_right (4x4 * 4x3) with a similar approach.
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # Given matrices X, Y (2x2 each), compute Z = X * Y using 7 products:
    # M1 = (X11 + X22) * (Y11 + Y22)
    # M2 = (X21 + X22) * Y11
    # M3 = X11 * (Y12 - Y22)
    # M4 = X22 * (Y21 - Y11)
    # M5 = (X11 + X12) * Y22
    # M6 = (X21 - X11) * (Y11 + Y12)
    # M7 = (X12 - X22) * (Y21 + Y22)
    #
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    def strassen_2x2_decomp():
        """
        Returns U, V, W factors for 2x2 * 2x2 = 2x2 multiplication.
        Each factor is 4 x 7 (for U, V) or 4 x 7 (for W).
        Convention: 
          A = [[a11, a12], [a21, a22]], indexed as A[0]=a11, A[1]=a12, A[2]=a21, A[3]=a22
          B = [[b11, b12], [b21, b22]], indexed as B[0]=b11, B[1]=b12, B[2]=b21, B[3]=b22
          C = [[c11, c12], [c21, c22]], indexed as C[0]=c11, C[1]=c12, C[2]=c21, C[3]=c22
        
        U is 4x7, V is 4x7, W is 4x7.
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (X11 + X22) * (Y11 + Y22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # Z11 += M1, Z22 += M1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (X21 + X22) * Y11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # Z21 += M2, Z22 -= M2
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = X11 * (Y12 - Y22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # Z12 += M3, Z22 += M3
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = X22 * (Y21 - Y11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # Z11 += M4, Z21 += M4
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (X11 + X12) * Y22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # Z11 -= M5, Z12 += M5
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (X21 - X11) * (Y11 + Y12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # Z22 += M6
        W[3, 5] = 1
        
        # M7 = (X12 - X22) * (Y21 + Y22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # Z11 += M7
        W[0, 6] = 1
        
        return U, V, W
    
    # Now build the full 4x4 * 4x4 decomposition using Strassen on 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2.
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # This is a 2x2 block matrix multiplication where each "element" is a 2x2 matrix.
    # We can apply Strassen at the block level, but the "multiplications" are 2x2*2x2.
    # So we use 7 "block multiplications", each of which is a 2x2*2x2 Strassen.
    # Total rank = 7 * 7 = 49.
    
    # For 4x4 * 4x4:
    # A indices: A11 = A[0:4], A12 = A[4:8], A21 = A[8:12], A22 = A[12:16] (row-major 2x2 blocks)
    # Actually, let's use the standard row-major indexing:
    # A is 4x4, stored as A[row*4+col] for row 0..3, col 0..3.
    # A11 = A[0:4] (rows 0-1, cols 0-1) = A[0], A[1], A[4], A[5] in flat index
    # A12 = A[2:6] (rows 0-1, cols 2-3) = A[2], A[3], A[6], A[7]
    # A21 = A[8:12] (rows 2-3, cols 0-1) = A[8], A[9], A[12], A[13]
    # A22 = A[10:14] (rows 2-3, cols 2-3) = A[10], A[11], A[14], A[15]
    
    # Let's define the Strassen 4x4 decomposition explicitly.
    def strassen_4x4_decomp():
        """
        Returns U (16x49), V (16x49), W (16x49) for 4x4 * 4x4 = 4x4.
        """
        U, V, W = np.zeros((16, 49)), np.zeros((16, 49)), np.zeros((16, 49))
        sU, sV, sW = strassen_2x2_decomp()
        
        # Block level Strassen:
        # S1 = (A11 + A22)
        # S2 = (A21 + A22)
        # S3 = A11
        # S4 = A22
        # S5 = (A11 + A12)
        # S6 = (A21 - A11)
        # S7 = (A12 - A22)
        #
        # T1 = (B11 + B22)
        # T2 = B11
        # T3 = (B12 - B22)
        # T4 = (B21 - B11)
        # T5 = B22
        # T6 = (B11 + B12)
        # T7 = (B21 + B22)
        #
        # M1 = S1 * T1
        # M2 = S2 * T2
        # M3 = S3 * T3
        # M4 = S4 * T4
        # M5 = S5 * T5
        # M6 = S6 * T6
        # M7 = S7 * T7
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A block indices (flat, row-major):
        # A11: [0, 1, 4, 5]
        # A12: [2, 3, 6, 7]
        # A21: [8, 9, 12, 13]
        # A22: [10, 11, 14, 15]
        
        # B block indices:
        # B11: [0, 1, 4, 5]
        # B12: [2, 3, 6, 7]
        # B21: [8, 9, 12, 13]
        # B22: [10, 11, 14, 15]
        
        # C block indices:
        # C11: [0, 1, 4, 5]
        # C12: [2, 3, 6, 7]
        # C21: [8, 9, 12, 13]
        # C22: [10, 11, 14, 15]
        
        A11 = [0, 1, 4, 5]
        A12 = [2, 3, 6, 7]
        A21 = [8, 9, 12, 13]
        A22 = [10, 11, 14, 15]
        
        B11 = [0, 1, 4, 5]
        B12 = [2, 3, 6, 7]
        B21 = [8, 9, 12, 13]
        B22 = [10, 11, 14, 15]
        
        C11 = [0, 1, 4, 5]
        C12 = [2, 3, 6, 7]
        C21 = [8, 9, 12, 13]
        C22 = [10, 11, 14, 15]
        
        # For each Strassen block multiplication, we use the 2x2 Strassen decomposition.
        # The block U factor maps A blocks to intermediate values.
        # We need to build U such that U[i, col] gives the coefficient for A[i] in the i-th intermediate.
        
        # Let's build this more carefully.
        # For each of the 7 block Strassen products, we have a 2x2 * 2x2 Strassen decomposition.
        # The U factor for each block product is a 4x7 matrix (for the 4 elements of the A-block).
        # We need to map these to the full 16x49 U matrix.
        
        # For block product j (j=0..6), the A-block is one of S1..S7.
        # S1 = A11 + A22: U_block = sU with contributions from A11 and A22.
        # S2 = A21 + A22: U_block from A21 and A22.
        # etc.
        
        # Let's define the A-block for each Strassen step:
        # Step 1: A_block = A11 + A22
        # Step 2: A_block = A21 + A22
        # Step 3: A_block = A11
        # Step 4: A_block = A22
        # Step 5: A_block = A11 + A12
        # Step 6: A_block = A21 - A11
        # Step 7: A_block = A12 - A22
        
        A_blocks = [
            (A11, 1, A22, 1),   # S1 = A11 + A22
            (A21, 1, A22, 1),   # S2 = A21 + A22
            (A11, 1, None, 0),  # S3 = A11
            (A22, 1, None, 0),  # S4 = A22
            (A11, 1, A12, 1),   # S5 = A11 + A12
            (A21, 1, A11, -1),  # S6 = A21 - A11
            (A12, 1, A22, -1),  # S7 = A12 - A22
        ]
        
        B_blocks = [
            (B11, 1, B22, 1),   # T1 = B11 + B22
            (B11, 1, None, 0),  # T2 = B11
            (B12, 1, B22, -1),  # T3 = B12 - B22
            (B21, 1, B11, -1),  # T4 = B21 - B11
            (B22, 1, None, 0),  # T5 = B22
            (B11, 1, B12, 1),   # T6 = B11 + B12
            (B21, 1, B22, 1),   # T7 = B21 + B22
        ]
        
        # C output contributions:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        C_contribs = [
            (C11, 1, C12, 0, C21, 0, C22, 1),  # M1: C11+=, C22+=
            (C11, 0, C12, 0, C21, 1, C22, -1), # M2: C21+=, C22-=
            (C11, 0, C12, 1, C21, 0, C22, 1),  # M3: C12+=, C22+=
            (C11, 1, C12, 0, C21, 1, C22, 0),  # M4: C11+=, C21+=
            (C11, -1, C12, 1, C21, 0, C22, 0), # M5: C11-=, C12+=
            (C11, 0, C12, 0, C21, 0, C22, 1),  # M6: C22+=
            (C11, 1, C12, 0, C21, 0, C22, 0),  # M7: C11+=
        ]
        
        for j in range(7):
            col_offset = j * 7
            
            # Build U for this block
            for r in range(7):
                # sU[:, r] gives the coefficients for the 4 elements of the A-block
                # We need to map these to the full A vector.
                a1_idx, a1_coeff, a2_idx, a2_coeff = A_blocks[j]
                for e in range(4):
                    val = sU[e, r]
                    if a1_idx is not None:
                        U[a1_idx[e], col_offset + r] += val * a1_coeff
                    if a2_idx is not None:
                        U[a2_idx[e], col_offset + r] += val * a2_coeff
            
            # Build V for this block
            b1_idx, b1_coeff, b2_idx, b2_coeff = B_blocks[j]
            for r in range(7):
                for e in range(4):
                    val = sV[e, r]
                    if b1_idx is not None:
                        V[b1_idx[e], col_offset + r] += val * b1_coeff
                    if b2_idx is not None:
                        V[b2_idx[e], col_offset + r] += val * b2_coeff
            
            # Build W for this block
            c11, c11_coeff, c12, c12_coeff, c21, c21_coeff, c22, c22_coeff = C_contribs[j]
            for r in range(7):
                for e in range(4):
                    val = sW[e, r]
                    if c11 is not None and c11_coeff != 0:
                        W[c11[e], col_offset + r] += val * c11_coeff
                    if c12 is not None and c12_coeff != 0:
                        W[c12[e], col_offset + r] += val * c12_coeff
                    if c21 is not None and c21_coeff != 0:
                        W[c21[e], col_offset + r] += val * c21_coeff
                    if c22 is not None and c22_coeff != 0:
                        W[c22[e], col_offset + r] += val * c22_coeff
        
        return U, V, W
    
    # Now for the 4x4 * 4x7 case:
    # B is 4x7. Split into B_left (4x4) and B_right (4x3).
    # C = A * B_left + A * B_right
    
    # For A * B_left (4x4 * 4x4): use Strassen 4x4, rank 49.
    # For A * B_right (4x4 * 4x3): 
    #   B_right is 4x3. We can pad it to 4x4 by adding a zero column.
    #   Then use Strassen 4x4, but only the first 3 columns of the result are needed.
    #   However, this would give rank 49 for the padded version, but we only need 3 columns.
    #   Alternatively, we can use a more efficient decomposition for 4x4 * 4x3.
    
    # Let's think about 4x4 * 4x3:
    # A is 4x4, B_right is 4x3.
    # We can split A into 2x2 blocks, and B_right into 2x? blocks.
    # B_right columns 0-1 form a 4x2 matrix, column 2 is a 4x1 vector.
    
    # Actually, let's just use the Strassen 4x4 approach for both parts.
    # For A * B_right, we pad B_right to 4x4 with a zero 4th column.
    # Then the result is 4x4, but we only take the first 3 columns.
    # The rank would be 49 for this part as well.
    # Total rank = 49 + 49 = 98.
    
    # Let's build this.
    U_left, V_left, W_left = strassen_4x4_decomp()
    
    # For the right part (4x4 * 4x3):
    # B_right is 4x3, stored as B_right[row*3+col] for row 0..3, col 0..2.
    # We pad to 4x4: B_padded[row*4+col] = B_right[row*3+col] for col 0..2, 0 for col 3.
    # The result C_right is 4x4, but we only use columns 0..2.
    # C_right[row*4+col] for col 0..2 are the actual outputs.
    
    # U_right = U_left (same A)
    # V_right: map from B_padded (16 elements) to the 49 intermediate values.
    #   B_padded indices:
    #   B_padded[0] = B_right[0] (row 0, col 0)
    #   B_padded[1] = B_right[1] (row 0, col 1)
    #   B_padded[2] = B_right[2] (row 0, col 2)
    #   B_padded[3] = 0 (row 0, col 3)
    #   B_padded[4] = B_right[3] (row 1, col 0)
    #   B_padded[5] = B_right[4] (row 1, col 1)
    #   B_padded[6] = B_right[5] (row 1, col 2)
    #   B_padded[7] = 0 (row 1, col 3)
    #   B_padded[8] = B_right[6] (row 2, col 0)
    #   B_padded[9] = B_right[7] (row 2, col 1)
    #   B_padded[10] = B_right[8] (row 2, col 2)
    #   B_padded[11] = 0 (row 2, col 3)
    #   B_padded[12] = B_right[9] (row 3, col 0)
    #   B_padded[13] = B_right[10] (row 3, col 1)
    #   B_padded[14] = B_right[11] (row 3, col 2)
    #   B_padded[15] = 0 (row 3, col 3)
    
    # V_right is a 12x49 matrix (B_right has 12 elements).
    # For each column j of V_left (which corresponds to B_padded indices 0..15),
    # we need to map to B_right indices.
    
    V_right = np.zeros((12, 49))
    # Mapping from B_padded index to B_right index, or -1 if zero
    bmap = [0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1]
    for i in range(16):
        if bmap[i] != -1:
            V_right[bmap[i], :] = V_left[i, :]
    
    # W_right: map from the 49 intermediate values to C_right (4x4), but only cols 0..2.
    # C_right has 12 elements (4 rows * 3 cols).
    # C_right[row*3+col] = C_padded[row*4+col] for col 0..2.
    W_right = np.zeros((12, 49))
    # Mapping from C_padded index to C_right index, or -1 if not used
    cmap = [0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1]
    for i in range(16):
        if cmap[i] != -1:
            W_right[cmap[i], :] = W_left[i, :]
    
    # Now combine:
    # U_total = [U_left, U_right] (16 x 98)
    # V_total = [V_left; V_right] (28 x 98) where V_left is 16x49 (B_left) and V_right is 12x49 (B_right)
    # W_total = [W_left; W_right] (28 x 98) where W_left is 16x49 (C_left, 4x4) and W_right is 12x49 (C_right, 4x3)
    
    # Wait, the convention for W is W[k*n+i, r] where k is col of C, i is row of C.
    # C is 4x7. So W has 7*4 = 28 rows.
    # C_left is 4x4 (cols 0-3 of C), C_right is 4x3 (cols 4-6 of C).
    # W_left maps to C_left: rows 0..15 (4 rows * 4 cols)
    # W_right maps to C_right: rows 16..27 (4 rows * 3 cols)
    
    # But W_left is 16x49 (for 4x4 C), W_right is 12x49 (for 4x3 C).
    # Total W should be 28x98.
    
    # Let's verify the W indexing.
    # W[k*n+i, r] where k = col index (0..6), i = row index (0..3).
    # So W has 28 rows: row k*4+i.
    
    # W_left corresponds to C_left (cols 0-3).
    # C_left[row, col] for row 0..3, col 0..3.
    # In the 4x4 Strassen, C is indexed as C[row*4+col].
    # So W_left[row*4+col, j] maps to C_left[row, col].
    # In the full W, this is W[col*4+row, j] for col 0..3.
    
    # Hmm, the convention is W[k*n+i, r] where k is col, i is row.
    # So for C_left[row, col], the W index is col*4 + row.
    
    # In the Strassen 4x4, C is indexed as C[row*4+col] (row-major).
    # So C[row, col] is at index row*4+col.
    # But W uses col*4+row.
    # So we need to transpose the W indices.
    
    # Let me re-examine. The task says "output factor follows the official trace convention W[k*n+i,r]".
    # So W[k*n+i, r] where k is the column index of C, i is the row index of C.
    # C is n x p = 4 x 7.
    # So W has p*n = 28 rows.
    
    # For C_left (4x4, cols 0-3):
    # C_left[i, k] for i=0..3, k=0..3.
    # W index = k*4 + i.
    
    # In Strassen 4x4, the C matrix is 4x4, with C[row, col].
    # C[row, col] corresponds to C_left[row, col].
    # So W_left should have W_left[col*4+row, j] = value.
    
    # But in my strassen_4x4_decomp, W is built with W[c11[e], col_offset+r] etc.
    # c11 = [0, 1, 4, 5] which are indices for C11 block in row-major: C11[0]=C[0,0], C11[1]=C[0,1], C11[2]=C[1,0], C11[3]=C[1,1].
    # So W[c11[e], j] = W[C_index, j] where C_index is row-major.
    # But we need W[col*4+row, j].
    # So we need to transpose the W indices.
    
    # Let's fix this. I'll create a helper to transpose the W factor from row-major to the required convention.
    
    def transpose_w(W_rowmajor, n, p):
        """
        W_rowmajor is (n*p, rank) with indices row*p+col (row-major for n x p matrix).
        We want W_transposed with indices col*n+row.
        """
        rank = W_rowmajor.shape[1]
        W_t = np.zeros((p*n, rank))
        for row in range(n):
            for col in range(p):
                W_t[col*n + row, :] = W_rowmajor[row*p + col, :]
        return W_t
    
    # For the 4x4 * 4x4 part:
    W_left_t = transpose_w(W_left, 4, 4)
    
    # For the 4x4 * 4x3 part:
    # C_right is 4x3. In row-major, C_right[row, col] is at index row*3+col.
    # But W_right was built with indices from the 4x4 Strassen, mapped to 4x3.
    # Let me rebuild W_right properly.
    
    # Actually, let's just build the full decomposition from scratch with the correct W convention.
    
    # Let me redo this more carefully.
    
    # For the full 4x4 * 4x7 decomposition:
    # U: 16 x rank (A has 16 elements)
    # V: 28 x rank (B has 16 elements: 4x7)
    # W: 28 x rank (C has 28 elements: 4x7)
    
    # Wait, V should be (m*p, rank) = (4*7, rank) = (28, rank).
    # B is 4x7, so B has 28 elements.
    # In row-major, B[row, col] is at index row*7+col.
    
    # For the left part (B_left, 4x4):
    # B_left[row, col] for row 0..3, col 0..3.
    # B index = row*7+col for col 0..3.
    
    # For the right part (B_right, 4x3):
    # B_right[row, col] for row 0..3, col 0..2.
    # B index = row*7+4+col for col 0..2.
    
    # Let's rebuild V properly.
    
    # V_left maps from B_left (16 elements) to 49 intermediates.
    # B_left[row, col] is at B index row*7+col.
    # In the Strassen 4x4, B is 4x4, B[row, col] at index row*4+col.
    
    V_left_full = np.zeros((28, 49))
    for row in range(4):
        for col in range(4):
            V_left_full[row*7+col, :] = V_left[row*4+col, :]
    
    # V_right maps from B_right (12 elements) to 49 intermediates.
    # B_right[row, col] for row 0..3, col 0..2.
    # B index = row*7+4+col.
    # In the padded 4x4, B_padded[row, col] for col 0..2 is at index row*4+col.
    
    V_right_full = np.zeros((28, 49))
    for row in range(4):
        for col in range(3):
            V_right_full[row*7+4+col, :] = V_right[row*3+col, :]
    
    # Now W:
    # W_left_t is 16x49, with W_left_t[col*4+row, j] for C_left[row, col].
    # C_left[row, col] for col 0..3.
    # Full C index = col*4+row.
    
    # W_right: C_right[row, col] for row 0..3, col 0..2.
    # Full C index = (4+col)*4+row = 16+col*4+row.
    
    # Let's build W_right_full properly.
    # From the Strassen 4x4, C_padded[row, col] for col 0..2.
    # In the 4x4 Strassen, C is 4x4, C[row, col] at index row*4+col.
    # So C_padded[row, col] is at index row*4+col for col 0..2.
    
    W_right_full = np.zeros((28, 49))
    for row in range(4):
        for col in range(3):
            # C_padded[row, col] is at index row*4+col in the 4x4 Strassen.
            # We need to get the value from W_left[row*4+col, :] (since W_right was derived from W_left).
            # Wait, W_right was built from W_left with a mapping. Let me just use W_left directly.
            W_right_full[(4+col)*4+row, :] = W_left[row*4+col, :]
    
    # Now combine:
    U_total = np.hstack([U_left, U_left])  # 16 x 98
    V_total = np.hstack([V_left_full, V_right_full])  # 28 x 98
    W_total = np.hstack([W_left_t, W_right_full])  # 28 x 98
    
    # Convert to lists of Fraction or int
    # The values are integers (from Strassen's algorithm).
    
    U_out = U_total.tolist()
    V_out = V_total.tolist()
    W_out = W_total.tolist()
    
    # Convert to exact fractions if needed, but they should be integers.
    # Let's verify they are integers.
    if not np.allclose(U_total, np.round(U_total)):
        # Convert to fractions
        U_out = [[Fraction(int(x)) if x == int(x) else Fraction(x).limit_denominator(1000) for x in row] for row in U_total]
    if not np.allclose(V_total, np.round(V_total)):
        V_out = [[Fraction(int(x)) if x == int(x) else Fraction(x).limit_denominator(1000) for x in row] for row in V_total]
    if not np.allclose(W_total, np.round(W_total)):
        W_out = [[Fraction(int(x)) if x == int(x) else Fraction(x).limit_denominator(1000) for x in row] for row in W_total]
    
    # Ensure they are integers if they are close to integers
    U_out = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in U_out]
    V_out = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in V_out]
    W_out = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
    print(f"Rank: {len(result['U'][0])}")
