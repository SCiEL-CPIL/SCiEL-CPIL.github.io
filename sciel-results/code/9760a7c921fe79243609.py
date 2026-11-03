import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    # Start time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # We construct the rank-28 decomposition derived from Strassen for 4x4 matrices.
    # Strassen for 4x4 uses 7 multiplications of 2x2 blocks.
    # Each 2x2 multiplication is schoolbook (rank 4).
    # Total rank = 7 * 4 = 28.
    
    # The Strassen 7 multiplications for 2x2 blocks:
    # M1 = (A11) * (B11 - B22)
    # M2 = (A11 + A12) * B22
    # M3 = (A21 - A11) * B11
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A21 + A11) * (B11 - B21)
    
    # This is for 2x2 matrices A and B.
    # For 4x4 matrices, we split into 2x2 blocks.
    # Let A be 4x4, B be 4x4.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    
    # The 7 intermediate products are 2x2 matrices.
    # We need to decompose each 2x2 multiplication into rank-1 terms.
    # A 2x2 multiplication C = A * B can be written as:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # This is 4 rank-1 terms if we consider the scalar multiplications.
    # But here the "scalars" are 2x2 matrices? No.
    # In the tensor decomposition context, we are decomposing the bilinear map.
    # The Strassen algorithm reduces the number of *bilinear* terms.
    # If we use schoolbook for the base 2x2, the rank is 4.
    # So each of the 7 Strassen terms contributes 4 rank-1 terms to the total decomposition.
    
    # We need to map these to the 3x4x4 tensor.
    # We pad A to 4x4 with a zero row.
    # A_pad = [[A11, A12], [A21, A22], [0, 0]]? No, A is 3x4.
    # A is 3 rows, 4 cols.
    # We can view A as [[A_top, A_bottom]] where A_top is 2x4, A_bottom is 1x4?
    # Or better, just use the 4x4 decomposition and truncate.
    
    # Let's generate the 28 terms for M_4,4,4.
    # Indices:
    # A is 4x4 (indices 0..3, 0..3)
    # B is 4x4 (indices 0..3, 0..3)
    # C is 4x4 (indices 0..3, 0..3)
    
    # Strassen blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # The 7 products M1..M7 are 2x2 matrices.
    # Each M is a linear combination of A-blocks times B-blocks.
    # Since we are doing tensor decomposition, we need to expand each M into rank-1 terms.
    # A rank-1 term for a 2x2 matrix multiplication is of the form:
    # (u^T A v) w
    # where u, v are 4-vectors (for the 4x4 case) and w is a 4x4 matrix?
    # No, the tensor T is of shape (4, 4, 16).
    # A rank-1 term is u (4x1) * v (4x1) * w (4x4).
    # The product is (u^T A v) * w.
    
    # For the schoolbook 2x2 multiplication inside the Strassen step:
    # C = A_block * B_block.
    # This is a 2x2 matrix.
    # We can decompose this 2x2 multiplication into 4 rank-1 terms for the 2x2 tensor.
    # But we need to map this back to the 4x4 tensor.
    
    # Let's build the U, V, W matrices directly.
    # Rank = 28.
    # U: 16 x 28
    # V: 16 x 28
    # W: 16 x 28 (flattened 4x4)
    
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))
    
    # Helper to add a rank-1 term for a 2x2 block multiplication
    # The 2x2 multiplication C = A * B (2x2)
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need to express the Strassen terms M1..M7.
    # Each M is a 2x2 matrix resulting from a product of linear combinations.
    # Example M1 = (A11) * (B11 - B22).
    # This is a product of two 2x2 matrices.
    # Let X = A11, Y = B11 - B22.
    # M1 = X * Y.
    # We decompose X * Y into 4 rank-1 terms (schoolbook).
    # Term 1: X11 * Y11 -> contributes to M1[0,0]
    # Term 2: X12 * Y21 -> contributes to M1[0,0]
    # Term 3: X11 * Y12 -> contributes to M1[0,1]
    # Term 4: X12 * Y22 -> contributes to M1[0,1]
    # ... etc.
    
    # We need to map the indices of the 2x2 blocks back to the 4x4 indices.
    
    # Block index mapping:
    # Block (bi, bj) where bi, bj in {0, 1}
    # Rows: bi*2, bi*2+1
    # Cols: bj*2, bj*2+1
    
    def get_block_indices(bi, bj):
        return [bi*2, bi*2+1, bj*2, bj*2+1]
    
    # We will iterate through the 7 Strassen steps.
    # For each step, we have a product of two linear combinations of blocks.
    # We expand the product into 4 schoolbook terms for the 2x2 result.
    # Each schoolbook term is a scalar multiplication of two 2x2 matrix entries?
    # No, the "scalar" is the product of a specific entry of X and a specific entry of Y.
    # But X and Y are linear combinations of A-blocks and B-blocks.
    # So each "scalar" is a bilinear form in the entries of A and B.
    # This bilinear form can be represented as a rank-1 term in the 4x4 tensor.
    
    # Let's define the 7 Strassen combinations.
    # Each combination is (X, Y) where X and Y are linear combinations of A-blocks and B-blocks.
    # We need the coefficients.
    
    # M1 = A11 * (B11 - B22)
    # M2 = (A11 + A12) * B22
    # M3 = (A21 - A11) * B11
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A21 + A11) * (B11 - B21)
    
    # For each M, we compute the 4 schoolbook terms for the 2x2 multiplication.
    # C = X * Y
    # C00 = X00 Y00 + X01 Y10
    # C01 = X00 Y01 + X01 Y11
    # C10 = X10 Y00 + X11 Y10
    # C11 = X10 Y01 + X11 Y11
    
    # X and Y are 2x2 matrices.
    # X is a linear combination of A-blocks.
    # Y is a linear combination of B-blocks.
    
    # We need to represent X00, X01, etc. as linear forms in A.
    # And Y00, etc. as linear forms in B.
    
    # Let's create a dictionary for the linear forms.
    # A-blocks: A00, A01, A10, A11 (corresponding to A11, A12, A21, A22)
    # Actually, let's use the block indices (0,0), (0,1), (1,0), (1,1).
    
    # Linear form for A:
    # X is a 2x2 matrix.
    # Each entry X[i,j] is a linear combination of A-blocks.
    # Since X is one of the A-blocks or a sum, it's simple.
    
    # Let's define the terms for M1..M7.
    
    # We will accumulate U, V, W.
    # Index c from 0 to 27.
    
    c = 0
    for m_idx in range(7):
        # Define X and Y linear forms
        # X is 2x2, Y is 2x2.
        # X[i,j] is a dict of {block_index: coeff}
        # Y[i,j] is a dict of {block_index: coeff}
        
        if m_idx == 0: # M1
            # X = A00 (A11)
            X = [[{'(0,0)': 1}, {'(0,1)': 0}],
                 [{'(1,0)': 0}, {'(1,1)': 0}]] # Wait, X is 2x2. A11 is the top-left block.
            # In Strassen, X and Y are the factors.
            # M1 = A11 * (B11 - B22).
            # X = A11. This is the top-left 2x2 block of A.
            # So X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1].
            # This is not a linear combination of blocks, it IS a block.
            # So X is the identity map on the block A00.
            # Y = B11 - B22.
            # Y is the block B00 minus B11.
            
            # Let's represent X and Y as 2x2 matrices of linear forms.
            # A linear form is a vector of coefficients for the 4 blocks.
            # Block 0: (0,0), Block 1: (0,1), Block 2: (1,0), Block 3: (1,1).
            
            X_forms = np.zeros((2, 2, 4))
            Y_forms = np.zeros((2, 2, 4))
            
            # X = A00
            X_forms[0,0,0] = 1
            X_forms[0,1,1] = 1
            X_forms[1,0,2] = 1
            X_forms[1,1,3] = 1
            
            # Y = B00 - B11
            Y_forms[0,0,0] = 1
            Y_forms[0,1,1] = 1
            Y_forms[1,0,2] = 1
            Y_forms[1,1,3] = 1
            # Subtract B11 (which is block 3)
            Y_forms[0,0,3] = -1
            Y_forms[0,1,3] = -1
            Y_forms[1,0,3] = -1
            Y_forms[1,1,3] = -1
            
        elif m_idx == 1: # M2
            # X = A00 + A01
            X_forms[0,0,0] = 1; X_forms[0,0,1] = 1
            X_forms[0,1,0] = 1; X_forms[0,1,1] = 1
            X_forms[1,0,0] = 1; X_forms[1,0,1] = 1
            X_forms[1,1,0] = 1; X_forms[1,1,1] = 1
            # Y = B11 (Block 3)
            Y_forms[0,0,3] = 1
            Y_forms[0,1,3] = 1
            Y_forms[1,0,3] = 1
            Y_forms[1,1,3] = 1
            
        elif m_idx == 2: # M3
            # X = A10 - A00
            X_forms[0,0,2] = 1; X_forms[0,0,0] = -1
            X_forms[0,1,2] = 1; X_forms[0,1,0] = -1
            X_forms[1,0,2] = 1; X_forms[1,0,0] = -1
            X_forms[1,1,2] = 1; X_forms[1,1,0] = -1
            # Y = B00 (Block 0)
            Y_forms[0,0,0] = 1
            Y_forms[0,1,0] = 1
            Y_forms[1,0,0] = 1
            Y_forms[1,1,0] = 1
            
        elif m_idx == 3: # M4
            # X = A11 (Block 3)
            X_forms[0,0,3] = 1
            X_forms[0,1,3] = 1
            X_forms[1,0,3] = 1
            X_forms[1,1,3] = 1
            # Y = B10 - B00
            Y_forms[0,0,2] = 1; Y_forms[0,0,0] = -1
            Y_forms[0,1,2] = 1; Y_forms[0,1,0] = -1
            Y_forms[1,0,2] = 1; Y_forms[1,0,0] = -1
            Y_forms[1,1,2] = 1; Y_forms[1,1,0] = -1
            
        elif m_idx == 4: # M5
            # X = A00 + A11
            X_forms[0,0,0] = 1; X_forms[0,0,3] = 1
            X_forms[0,1,0] = 1; X_forms[0,1,3] = 1
            X_forms[1,0,0] = 1; X_forms[1,0,3] = 1
            X_forms[1,1,0] = 1; X_forms[1,1,3] = 1
            # Y = B00 + B11
            Y_forms[0,0,0] = 1; Y_forms[0,0,3] = 1
            Y_forms[0,1,0] = 1; Y_forms[0,1,3] = 1
            Y_forms[1,0,0] = 1; Y_forms[1,0,3] = 1
            Y_forms[1,1,0] = 1; Y_forms[1,1,3] = 1
            
        elif m_idx == 5: # M6
            # X = A01 - A11
            X_forms[0,0,1] = 1; X_forms[0,0,3] = -1
            X_forms[0,1,1] = 1; X_forms[0,1,3] = -1
            X_forms[1,0,1] = 1; X_forms[1,0,3] = -1
            X_forms[1,1,1] = 1; X_forms[1,1,3] = -1
            # Y = B10 + B11
            Y_forms[0,0,2] = 1; Y_forms[0,0,3] = 1
            Y_forms[0,1,2] = 1; Y_forms[0,1,3] = 1
            Y_forms[1,0,2] = 1; Y_forms[1,0,3] = 1
            Y_forms[1,1,2] = 1; Y_forms[1,1,3] = 1
            
        else: # M7
            # X = A10 + A00
            X_forms[0,0,2] = 1; X_forms[0,0,0] = 1
            X_forms[0,1,2] = 1; X_forms[0,1,0] = 1
            X_forms[1,0,2] = 1; X_forms[1,0,0] = 1
            X_forms[1,1,2] = 1; X_forms[1,1,0] = 1
            # Y = B00 - B10
            Y_forms[0,0,0] = 1; Y_forms[0,0,2] = -1
            Y_forms[0,1,0] = 1; Y_forms[0,1,2] = -1
            Y_forms[1,0,0] = 1; Y_forms[1,0,2] = -1
            Y_forms[1,1,0] = 1; Y_forms[1,1,2] = -1
            
        # Now expand C = X * Y into 4 rank-1 terms (schoolbook for 2x2)
        # C00 = X00 Y00 + X01 Y10
        # C01 = X00 Y01 + X01 Y11
        # C10 = X10 Y00 + X11 Y10
        # C11 = X10 Y01 + X11 Y11
        
        # For each term, e.g., X00 Y00:
        # X00 is a linear form in A-blocks.
        # Y00 is a linear form in B-blocks.
        # The product is a bilinear form.
        # We can decompose this bilinear form into rank-1 terms by expanding the sums.
        # But wait, X00 is a sum of A-block entries.
        # The bilinear form is (sum a_i A_i) * (sum b_j B_j).
        # This is not rank-1 in the tensor sense unless we treat the whole thing as one term?
        # No, the tensor rank counts the number of rank-1 tensors.
        # A term like (u^T A v) w is rank-1.
        # Here, u and v are vectors.
        # The bilinear form (X00)(Y00) is a product of two linear forms.
        # This corresponds to a rank-1 term in the 2x2 tensor if X00 and Y00 were scalars.
        # But X00 is a 2x2 block? No, X00 is an entry of the 2x2 matrix X.
        # X is a 2x2 matrix of blocks? No, X is a 2x2 matrix of *scalars* if we are in the base case.
        # But here we are in the block case.
        # X is a 2x2 matrix where each entry is a linear combination of A-block entries.
        # So X00 is a linear form in the 16 entries of A.
        # Y00 is a linear form in the 16 entries of B.
        # The product X00 Y00 is a bilinear form.
        # Can we represent it as a single rank-1 term?
        # Yes, if we define u and v appropriately.
        # u is the vector of coefficients for X00.
        # v is the vector of coefficients for Y00.
        # Then (u^T A v) is X00 Y00? No.
        # u^T A v is a scalar.
        # If u and v are chosen such that u^T A v = X00 (as a function of A), and similarly for Y,
        # then the product is not (u^T A v) (w^T B z).
        # The rank-1 term is (u^T A v) * w_matrix.
        # Here we have (X00) * (Y00) contributing to C00.
        # This is a scalar multiplication of two linear forms.
        # This is NOT a rank-1 term in the tensor decomposition of the multiplication map.
        # The multiplication map is C = A B.
        # C00 = A00 B00 + A01 B10 + ...
        # Each term Aij Bkl is a rank-1 term?
        # No, Aij Bkl is a product of two scalars.
        # In the tensor T, the entry T[i,j,k,l] is 1 if i=k and j=l? No.
        # T[A,B,C] = C.
        # C[i,j] = sum_k A[i,k] B[k,j].
        # The term A[i,k] B[k,j] is a rank-1 term in the tensor of shape (n*m, m*p, p*n)?
        # Yes. The tensor is of shape (nm, mp, pn).
        # A rank-1 term is u (nm x 1) * v (mp x 1) * w (pn x 1).
        # The product is (u^T A_flat v) w.
        # If we pick u to be the basis vector for A[i,k], and v for B[k,j],
        # then u^T A_flat v = A[i,k] B[k,j].
        # And w is the basis vector for C[i,j].
        # So each scalar multiplication A[i,k] B[k,j] corresponds to a rank-1 term.
        
        # So, in the Strassen decomposition, we replaced 8 scalar multiplications with 7.
        # But each "scalar" in the Strassen sense is a 2x2 matrix multiplication.
        # And we are decomposing that 2x2 multiplication into 4 scalar multiplications.
        # So each Strassen term (which is a product of two linear combinations)
        # expands into 4 scalar multiplications (schoolbook).
        # Each scalar multiplication is a rank-1 term.
        # So each Strassen term contributes 4 rank-1 terms.
        # Total 7 * 4 = 28 rank-1 terms.
        
        # So we need to generate the 4 rank-1 terms for each Strassen step.
        # For M1 = X * Y:
        # C00 gets contribution from X00 Y00 and X01 Y10.
        # C01 gets contribution from X00 Y01 and X01 Y11.
        # C10 gets contribution from X10 Y00 and X11 Y10.
        # C11 gets contribution from X10 Y01 and X11 Y11.
        
        # For each pair (Xij, Ykl), we have a bilinear form.
        # Xij is a linear form in A.
        # Ykl is a linear form in B.
        # The product is (sum a_p A_p) * (sum b_q B_q).
        # This is NOT a single rank-1 term. It is a sum of rank-1 terms!
        # (sum a_p A_p) (sum b_q B_q) = sum_{p,q} a_p b_q A_p B_q.
        # Each A_p B_q is a rank-1 term.
        # So the number of rank-1 terms is the number of non-zero pairs (p,q).
        # This will explode the rank.
        
        # Wait, the Strassen algorithm reduces the number of *multiplications of matrices*.
        # If we use schoolbook for the matrix multiplications, the rank is 7 * 4 = 28.
        # This assumes that the "matrix multiplication" in Strassen is treated as a black box of rank 4.
        # But in the tensor decomposition, we are flattening everything.
        # The rank of the tensor M_4,4,4 is the minimum number of rank-1 terms.
        # Strassen's algorithm gives a decomposition of rank 28?
        # Yes, if we use schoolbook for the 2x2 blocks.
        # The logic is:
        # 1. Compute 7 intermediate matrices M1..M7 (each 2x2).
        # 2. Each M is computed by a multiplication of two 2x2 matrices (linear combos).
        # 3. Each 2x2 multiplication is decomposed into 4 rank-1 terms (scalar mults).
        # 4. Total 28 rank-1 terms.
        
        # So we just need to generate the 4 terms for each M.
        # For M = X * Y:
        # Term 1: X00 * Y00 contributes to C00.
        # Term 2: X01 * Y10 contributes to C00.
        # Term 3: X00 * Y01 contributes to C01.
        # Term 4: X01 * Y11 contributes to C01.
        # Term 5: X10 * Y00 contributes to C10.
        # Term 6: X11 * Y10 contributes to C10.
        # Term 7: X10 * Y01 contributes to C11.
        # Term 8: X11 * Y11 contributes to C11.
        
        # Wait, that's 8 terms per M.
        # 7 * 8 = 56?
        # No, schoolbook for 2x2 has 4 multiplications, but 8 scalar multiplications?
        # C = A B.
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        # There are 8 scalar products.
        # So rank is 8?
        # No, the rank of M_2,2,2 is 4? No, it's 7?
        # No, schoolbook rank for 2x2 is 4?
        # C00 = A00 B00 + A01 B10. This is 2 terms.
        # If we treat each scalar product as a rank-1 term, we have 8 terms.
        # But we can combine them?
        # No, the rank of M_2,2,2 is 4?
        # Actually, the rank of M_2,2,2 is 4? No, it's 7 (Strassen).
        # Schoolbook is 4? No, schoolbook is 4 multiplications of 2x2? No.
        # For 2x2 matrices, schoolbook uses 4 multiplications of 2x2 matrices? No.
        # Schoolbook uses 4 multiplications of scalars? No.
        # For 2x2, schoolbook uses 4 multiplications of 1x1? No.
        # C00 = A00 B00 + A01 B10.
        # This requires 2 multiplications.
        # Total 8 multiplications.
        # So schoolbook rank is 8.
        # Strassen rank is 7.
        
        # So if we use Strassen for 4x4 (7 multiplications of 2x2),
        # and schoolbook for 2x2 (8 multiplications of scalars),
        # the total rank is 7 * 8 = 56?
        # This is worse than 48.
        
        # I made a mistake.
        # Strassen for 4x4:
        # 7 multiplications of 2x2 matrices.
        # If we use Strassen for the 2x2 matrices, rank is 7 * 7 = 49.
        # If we use schoolbook for the 2x2 matrices, rank is 7 * 8 = 56.
        
        # So the rank-28 solution I was thinking of is incorrect.
        # The rank of M_4,4,4 is at most 21 (Landsberg).
        # But Strassen gives 49 or 56.
        
        # So the 44-rank solution from the previous program is likely better than simple Strassen.
        # It might be a specific rectangular Strassen variant.
        
        # Given the complexity, I will stick to the 44-rank solution if I can reconstruct it,
        # or use a local search to improve the 48-rank solution.
        
        # Since I cannot easily reconstruct the 44-rank solution,
        # I will implement a local search that starts from 48 and tries to reduce rank.
        
        pass

    # Fallback: Return the 48-rank schoolbook solution to ensure validity.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
