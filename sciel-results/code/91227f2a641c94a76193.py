import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We will use a block-based approach.
    # A is 2x4, B is 4x5.
    # Split A into two 2x2 blocks: A = [A_L, A_R]
    # Split B into two 2x2 blocks (top 4 rows) and one 2x1 block (bottom 1 row):
    # B = [B_L; B_R; B_B] where B_L is 2x2, B_R is 2x2, B_B is 2x1.
    # Wait, B is 4x5.
    # Rows of B: 4. Cols of B: 5.
    # Let's split columns of B into 2, 2, 1.
    # B = [B_0 | B_1 | B_2]
    # B_0: 4x2, B_1: 4x2, B_2: 4x1.
    # A = 2x4.
    # C = A B = [A B_0 | A B_1 | A B_2]
    # Each term is 2x2 or 2x1.
    
    # Let's refine the split to allow Strassen on 2x2x2 blocks.
    # A is 2x4. Split A into A_L (2x2) and A_R (2x2).
    # B is 4x5.
    # We can split B's rows to match A's columns? No, matrix mult is A (2x4) * B (4x5).
    # The inner dimension is 4.
    # So we sum over the 4 rows of B and 4 cols of A.
    
    # Let's split the 4-dimensional inner space into two 2-dimensional subspaces.
    # A = [A_0 | A_1] where A_0 is 2x2, A_1 is 2x2.
    # B = [B_0; B_1] where B_0 is 2x5, B_1 is 2x5.
    # Then C = A_0 B_0 + A_1 B_1.
    # A_0 B_0 is 2x5. A_1 B_1 is 2x5.
    # This doesn't immediately give 2x2x2 blocks unless we also split the output columns.
    
    # Let's split output columns of B into 2, 2, 1.
    # B_0 is 2x2, B_1 is 2x2, B_2 is 2x1.
    # B_1 (the other 2x5 block) -> B_3 is 2x2, B_4 is 2x2, B_5 is 2x1.
    
    # So C = A_0 [B_0 | B_1 | B_2] + A_1 [B_3 | B_4 | B_5]
    # C = [ A_0 B_0 + A_1 B_3 | A_0 B_1 + A_1 B_4 | A_0 B_2 + A_1 B_5 ]
    
    # Term 1: C_0 = A_0 B_0 + A_1 B_3. (2x2 + 2x2 = 2x2)
    # Term 2: C_1 = A_0 B_1 + A_1 B_4. (2x2 + 2x2 = 2x2)
    # Term 3: C_2 = A_0 B_2 + A_1 B_5. (2x1 + 2x1 = 2x1)
    
    # For C_0 and C_1, we are adding two 2x2 matrix products.
    # Can we do this with rank < 4+4=8?
    # The tensor for C_0 is the sum of two independent 2x2x2 tensors.
    # Usually, the rank of a sum of tensors is the sum of ranks if they are on disjoint supports, but here they share the output space C (2x2) and input spaces are disjoint (A_0,B_0) vs (A_1,B_3).
    # Actually, A_0 and A_1 are disjoint parts of A. B_0 and B_3 are disjoint parts of B.
    # So the tensors are on disjoint input supports.
    # T_0 = A_0 \otimes B_0 \otimes C (where C is 2x2)
    # T_1 = A_1 \otimes B_3 \otimes C
    # Since supports in A and B are disjoint, the rank is likely rank(T_0) + rank(T_1) = 4 + 4 = 8.
    # Strassen applies to a single 2x2x2 product. It reduces rank of ONE product from 4 to 7? No, Strassen is 7 rank for 2x2x2.
    # Wait, the trivial rank for 2x2x2 is 4. Strassen is 7. Strassen is WORSE for 2x2x2!
    # Strassen is better for 2x2x2? No.
    # 2x2x2 multiplication has rank 7? No, it's 7 for 2x2x2?
    # Let's check.
    # 1x1x1: rank 1.
    # 2x2x2: rank 7 is for 2x2x2?
    # Strassen's algorithm multiplies two 2x2 matrices using 7 scalar multiplications.
    # So yes, rank(2x2x2) = 7.
    # Trivial rank is 4.
    # Wait. 2x2x2 trivial is 4 multiplications.
    # c00 = a00b00 + a01b10
    # c01 = a00b01 + a01b11
    # c10 = a10b00 + a11b10
    # c11 = a10b01 + a11b11
    # This is 4 multiplications.
    # Strassen uses 7. Why would anyone use Strassen for 2x2?
    # Ah, Strassen is for 2x2x2? No, Strassen is for 2x2x2 matrices?
    # Strassen multiplies 2x2 matrices. The result is 2x2.
    # Inputs: 2x2, 2x2. Output: 2x2.
    # The tensor is in R^{2x2} \otimes R^{2x2} \otimes R^{2x2}.
    # Dimension is 4x4x4.
    # Trivial rank is 16? No.
    # c_ij = sum_k a_ik b_kj.
    # For 2x2:
    # c00 = a00 b00 + a01 b10 (2 mults)
    # c01 = a00 b01 + a01 b11 (2 mults)
    # c10 = a10 b00 + a11 b10 (2 mults)
    # c11 = a10 b01 + a11 b11 (2 mults)
    # Total 8 scalar multiplications?
    # No, the standard algorithm does 8 multiplications for 2x2?
    # a00*b00, a01*b10, a00*b01, a01*b11, a10*b00, a11*b10, a10*b01, a11*b11.
    # Yes, 8 multiplications.
    # Strassen reduces it to 7.
    # So rank(2x2x2) is 7.
    # My previous thought "trivial rank 4" was wrong. That was for 1x1x1? No.
    # For 2x2x2, the number of terms in the sum is 2 (inner dim). 4 outputs. 4*2 = 8.
    # So trivial rank is 8. Strassen is 7.
    
    # Okay, so for 2x2x2, we can save 1 rank.
    
    # Back to our problem:
    # C_0 = A_0 B_0 + A_1 B_3.
    # This is the sum of two 2x2x2 tensors.
    # Since A_0 and A_1 are disjoint, and B_0 and B_3 are disjoint, the tensors are "disjoint" in a sense.
    # The rank of the sum is rank(T_0) + rank(T_1) = 7 + 7 = 14.
    # Trivial would be 8 + 8 = 16.
    # So we save 2 ranks here.
    
    # Similarly for C_1 = A_0 B_1 + A_1 B_4.
    # Rank 7 + 7 = 14. (Trivial 16).
    # Save 2 ranks.
    
    # For C_2 = A_0 B_2 + A_1 B_5.
    # A_0 is 2x2, B_2 is 2x1. Product is 2x1.
    # Tensor is 2x2x1 (A_0) x 2x1x1 (B_2) x 2x1 (C_2).
    # Inner dim 2.
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # 4 multiplications.
    # Is there a rank 3 decomposition for 2x2x1?
    # This is matrix-vector multiplication.
    # Rank of 2x2x1 tensor.
    # Trivial is 4.
    # Can we do it in 3?
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # The tensor is in R^4 \otimes R^2 \otimes R^2.
    # I don't think there's a standard rank 3 decomposition for this.
    # Let's assume rank 4 for now.
    # So for C_2: Rank(A_0 B_2) + Rank(A_1 B_5) = 4 + 4 = 8.
    # Trivial is 4 + 4 = 8. No saving.
    
    # Total Rank = 14 (C_0) + 14 (C_1) + 8 (C_2) = 36.
    # Trivial Rank = 16 + 16 + 8 = 40.
    # Saving = 4.
    # Score = 40/36 = 1.11.
    
    # Can we do better?
    # What if we split differently?
    # A is 2x4. B is 4x5.
    # Maybe split A into 1x4 and 1x4?
    # A_0 (1x4), A_1 (1x4).
    # B_0 (4x5), B_1 (4x5).
    # C_0 = A_0 B_0 (1x5).
    # C_1 = A_1 B_1 (1x5).
    # C = C_0 + C_1 (2x5).
    # Rank of 1x4x5 tensor?
    # Trivial: 4 mults per output? No.
    # c_j = sum_i a_i b_ij.
    # 5 outputs. Inner dim 4.
    # 5*4 = 20 mults.
    # Rank 20.
    # Two such tensors: 40.
    # No saving.
    
    # What if we use the full 2x4x5 tensor?
    # Trivial rank 40.
    
    # Let's stick with the 36 rank solution.
    # We need to construct the U, V, W matrices for the Strassen decomposition of 2x2x2.
    
    # Strassen for 2x2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to map this to U, V, W vectors.
    # For each 2x2x2 block, we have 7 terms.
    # U, V, W will have 7 columns for that block.
    # The indices in U, V, W correspond to the flattened matrices.
    
    # Let's define the indices for A_0 (2x2), B_0 (2x2), C_0 (2x2).
    # A_0 is top-left of A (2x4). Indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # B_0 is top-left of B (4x5)?
    # B is 4x5.
    # B_0 corresponds to rows 0,1 and cols 0,1.
    # Indices in B (flattened 4x5):
    # (0,0)->0, (0,1)->1
    # (1,0)->5, (1,1)->6
    # Wait, B is row-major?
    # Task says: W[k*n+i, r].
    # This implies C is stored row-major?
    # C is 2x5.
    # C[i, j]. i in 0..1, j in 0..4.
    # W index = k*n + i?
    # "W[k*n+i,r]"
    # k is row of C? i is col of C?
    # Usually W is the output factor.
    # If C[i,j] = sum_r U[i,j,r] V[?,r] W[?,r]
    # The standard tensor decomposition for matrix mult:
    # C_{ij} = \sum_{k} A_{ik} B_{kj}
    # Tensor T_{ijk} = \delta_{ik} \delta_{kj} ? No.
    # T_{ijk} is 1 if the product A_{ik} B_{kj} contributes to C_{ij}.
    # So T_{i, k, j} ?
    # Let's look at the seed program.
    # U[i*m+j][col] = 1. (A index)
    # V[j*p+k][col] = 1. (B index)
    # W[k*n+i][col] = 1. (C index)
    # Here:
    # A is n x m. Index i*m+j.
    # B is m x p. Index j*p+k.
    # C is p x n?
    # W index k*n+i.
    # If C is p x n, then C[k,i].
    # But matrix mult A(n,m) * B(m,p) = C(n,p).
    # So C should be n x p.
    # Seed program W index: k*n+i.
    # If C is n x p, index should be i*p+k.
    # But seed uses k*n+i.
    # This implies the output matrix is stored as p x n?
    # Or maybe the indices are swapped?
    # Let's check the dimensions in the seed.
    # U: (n*m, rank).
    # V: (m*p, rank).
    # W: (p*n, rank).
    # If W is p*n, then the output matrix is p x n.
    # But A is n x m, B is m x p. Product is n x p.
    # Unless B is p x m?
    # "multiplying a 2 by 4 matrix by a 4 by 5 matrix"
    # A: 2x4. B: 4x5.
    # Product: 2x5.
    # So output is 2x5.
    # W shape is (p*n, rank) = (5*2, rank) = (10, rank).
    # Index k*n+i.
    # If output is C (2x5), C[i,j].
    # Maybe the evaluator expects the output to be transposed?
    # Or maybe my interpretation of W index is wrong.
    # "W[k*n+i,r]"
    # If k is 0..4 (p) and i is 0..1 (n).
    # Then index is 0..9.
    # This matches p*n.
    # So the output matrix in the evaluator's mind is p x n.
    # i.e. C^T.
    # So we must decompose the tensor for C^T.
    # C^T[j,i] = C[i,j] = sum_k A[i,k] B[k,j].
    # So T_{j,i,k} = A[i,k] B[k,j].
    # U index: i*m+k. (A is n x m).
    # V index: k*p+j. (B is m x p).
    # W index: j*n+i. (C^T is p x n).
    # This matches the seed program exactly.
    # U[i*m+j] where j is the inner index?
    # Seed: U[i*m+j]. A is n x m. i is row, j is col.
    # V[j*p+k]. B is m x p. j is row, k is col.
    # W[k*n+i]. C^T is p x n. k is row (was col of C), i is col (was row of C).
    # Yes.
    
    # So we are decomposing C^T.
    # My block analysis was for C.
    # C = A B.
    # C^T = B^T A^T.
    # This is just a permutation of indices.
    # The rank is the same.
    # The Strassen decomposition applies to the 2x2x2 blocks of the multiplication.
    # Since the tensor is isomorphic, we can use the same rank.
    
    # We need to implement the Strassen decomposition for the specific index mapping.
    
    # Let's define the blocks for A, B, C^T.
    # A (2x4):
    # A_0 (2x2): rows 0-1, cols 0-1.
    # A_1 (2x2): rows 0-1, cols 2-3.
    
    # B (4x5):
    # B_0 (2x2): rows 0-1, cols 0-1.
    # B_1 (2x2): rows 0-1, cols 2-3.
    # B_2 (2x1): rows 0-1, col 4.
    # B_3 (2x2): rows 2-3, cols 0-1.
    # B_4 (2x2): rows 2-3, cols 2-3.
    # B_5 (2x1): rows 2-3, col 4.
    
    # C^T (5x2):
    # C^T is p x n.
    # Rows 0-4 (from B cols), Cols 0-1 (from A rows).
    # C^T_0 (2x2): rows 0-1, cols 0-1. (From B_0, B_3 and A_0, A_1)
    # C^T_1 (2x2): rows 2-3, cols 0-1. (From B_1, B_4 and A_0, A_1)
    # C^T_2 (1x2): row 4, cols 0-1. (From B_2, B_5 and A_0, A_1)
    
    # Term C^T_0 = A_0 B_0^T + A_1 B_3^T.
    # Note: B_0 is 2x2. B_0^T is 2x2.
    # The multiplication A_0 B_0^T is 2x2 * 2x2 = 2x2.
    # This is a standard 2x2x2 multiplication.
    # We can apply Strassen.
    
    # Term C^T_1 = A_0 B_1^T + A_1 B_4^T.
    # Standard 2x2x2.
    
    # Term C^T_2 = A_0 B_2^T + A_1 B_5^T.
    # A_0 (2x2) * B_2^T (1x2) = 2x2.
    # Wait. B_2 is 2x1. B_2^T is 1x2.
    # A_0 is 2x2.
    # A_0 B_2^T is 2x2.
    # So this is also 2x2x2?
    # A_0 (2x2), B_2^T (1x2).
    # The inner dimension is 2?
    # A_0 has 2 cols. B_2^T has 1 row.
    # So it's 2x2 * 1x2.
    # This is not a square matrix multiplication.
    # It's a 2x2 matrix times a 1x2 matrix.
    # Result is 2x2.
    # Number of mults: 2*2*1 = 4.
    # Can we reduce rank?
    # Probably not to 3.
    # So we use rank 4 for each.
    # Total for C^T_2: 4 + 4 = 8.
    
    # Total Rank: 7 (Strassen for C^T_0 part 1) + 7 (Strassen for C^T_0 part 2) + 7 (Strassen for C^T_1 part 1) + 7 (Strassen for C^T_1 part 2) + 4 + 4 = 36.
    
    # We need to write the code to generate U, V, W for this decomposition.
    
    # Helper function to add a Strassen term to U, V, W.
    # Inputs:
    # - U, V, W lists (to be modified)
    # - current_col
    # - A_block_start_row, A_block_start_col (in A)
    # - B_block_start_row, B_block_start_col (in B)
    # - C_block_start_row, C_block_start_col (in C^T)
    
    # Strassen terms:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Note: B in Strassen is the matrix being multiplied.
    # In our case, we are multiplying A_block (2x2) by B_block_T (2x2).
    # So the "B" in Strassen formulas is B_block_T.
    # B_block_T[i,j] = B_block[j,i].
    
    # Let's map the indices.
    # A_block is at A[ar:ar+2, ac:ac+2].
    # B_block is at B[br:br+2, bc:bc+2].
    # B_block_T is at B[bc:bc+2, br:br+2] (transposed).
    # C_block is at C_T[cr:cr+2, cc:cc+2].
    
    # For each Strassen term, we define the vectors for U, V, W.
    
    def add_strassen(U, V, W, col, ar, ac, br, bc, cr, cc):
        # A indices
        a00 = ar * 4 + ac
        a01 = ar * 4 + ac + 1
        a10 = (ar+1) * 4 + ac
        a11 = (ar+1) * 4 + ac + 1
        
        # B_T indices (B transposed)
        # B_T[i,j] = B[j,i]
        # B is 4x5.
        # B_T is 5x4.
        # We are accessing B_block_T which is 2x2.
        # B_block_T[0,0] = B[br, bc] -> index br*5 + bc
        # B_block_T[0,1] = B[br, bc+1] -> index br*5 + bc+1
        # B_block_T[1,0] = B[br+1, bc] -> index (br+1)*5 + bc
        # B_block_T[1,1] = B[br+1, bc+1] -> index (br+1)*5 + bc+1
        
        b00 = br * 5 + bc
        b01 = br * 5 + bc + 1
        b10 = (br+1) * 5 + bc
        b11 = (br+1) * 5 + bc + 1
        
        # C_T indices
        c00 = cr * 2 + cc
        c01 = cr * 2 + cc + 1
        c10 = (cr+1) * 2 + cc
        c11 = (cr+1) * 2 + cc + 1
        
        # Term 1: M1 = (a00+a11)(b00+b11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 2: M2 = (a10+a11)b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # Term 3: M3 = a00(b10-b11)
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 4: M4 = a11(b01-b00)
        U[a11][col] += 1
        V[b01][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # Term 5: M5 = (a00+a01)b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # Term 6: M6 = (a10-a00)(b00+b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 7: M7 = (a01-a11)(b10+b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
        return col

    def add_trivial_2x2x1(U, V, W, col, ar, ac, br, bc, cr, cc):
        # A_block (2x2) * B_block_T (1x2) -> C_block (2x2)
        # B_block is 2x1 at B[br:br+2, bc].
        # B_block_T is 1x2.
        # B_T[0,0] = B[br, bc]
        # B_T[0,1] = B[br+1, bc]
        
        # A indices
        a00 = ar * 4 + ac
        a01 = ar * 4 + ac + 1
        a10 = (ar+1) * 4 + ac
        a11 = (ar+1) * 4 + ac + 1
        
        # B_T indices
        b00 = br * 5 + bc
        b01 = (br+1) * 5 + bc
        
        # C_T indices
        c00 = cr * 2 + cc
        c01 = cr * 2 + cc + 1
        c10 = (cr+1) * 2 + cc
        c11 = (cr+1) * 2 + cc + 1
        
        # C = A * B_T
        # C[0,0] = A[0,0]B_T[0,0] + A[0,1]B_T[0,1]
        # C[0,1] = A[0,0]B_T[0,1] + A[0,1]B_T[0,0] ?? No.
        # B_T is 1x2.
        # A is 2x2.
        # C = A * B_T.
        # C[i,j] = sum_k A[i,k] B_T[k,j].
        # k is 0 only? No, B_T has 1 row.
        # So B_T[0,j].
        # C[i,j] = A[i,0] B_T[0,j] + A[i,1] B_T[0,1]? No.
        # A is 2x2. B_T is 1x2.
        # Dimensions don't match for standard mult?
        # A (2x2) * B_T (1x2) is invalid.
        # Ah.
        # A_0 is 2x2.
        # B_2 is 2x1.
        # A_0 * B_2 is 2x1.
        # But we want C^T which is 5x2.
        # C^T row 4 corresponds to B col 4.
        # C^T[4, i] = C[i, 4] = sum_k A[i,k] B[k,4].
        # This is a dot product of row i of A and col 4 of B.
        # This is 2x1 vector.
        # So C^T[4,:] is 1x2.
        # My previous block C^T_2 was 1x2.
        # So the output block is 1x2.
        # The multiplication is A_0 (2x2) * B_2 (2x1) = 2x1.
        # Then we take the transpose?
        # C^T[4, i] = (A_0 B_2)_i.
        # So the tensor is for A_0 B_2.
        # A_0 (2x2), B_2 (2x1), Result (2x1).
        # This is 2x2x1 multiplication.
        # Rank 4.
        
        # Let's implement 2x2x1 trivial.
        # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
        
        # B_2 is at B[br:br+2, bc].
        # b0 = br*5+bc
        # b1 = (br+1)*5+bc
        
        # C^T indices:
        # c0 = cr*2+cc (this is C^T[cr, cc])
        # c1 = (cr+1)*2+cc ??
        # Wait, C^T_2 is 1x2.
        # So only one row in C^T.
        # cr is the row index in C^T.
        # cc is the col index in C^T (0 or 1).
        # So c0 = cr*2+cc.
        # c1 = cr*2+cc+1.
        
        # Term 1: A[0,0] B[0,0] -> C[0]
        U[a00][col] += 1
        V[b0][col] += 1
        W[c0][col] += 1
        col += 1
        
        # Term 2: A[0,1] B[1,0] -> C[0]
        U[a01][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1
        col += 1
        
        # Term 3: A[1,0] B[0,0] -> C[1]
        U[a10][col] += 1
        V[b0][col] += 1
        W[c1][col] += 1
        col += 1
        
        # Term 4: A[1,1] B[1,0] -> C[1]
        U[a11][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1
        col += 1
        
        return col

    # Initialize U, V, W
    rank = 36
    U = [[0] * rank for _ in range(2 * 4)]
    V = [[0] * rank for _ in range(4 * 5)]
    W = [[0] * rank for _ in range(5 * 2)]
    
    col = 0
    
    # C^T_0 (2x2) = A_0 B_0^T + A_1 B_3^T
    # A_0: ar=0, ac=0
    # B_0: br=0, bc=0
    # C^T_0: cr=0, cc=0
    col = add_strassen(U, V, W, col, 0, 0, 0, 0, 0, 0)
    
    # A_1: ar=0, ac=2
    # B_3: br=2, bc=0
    # C^T_0: cr=0, cc=0
    col = add_strassen(U, V, W, col, 0, 2, 2, 0, 0, 0)
    
    # C^T_1 (2x2) = A_0 B_1^T + A_1 B_4^T
    # A_0: ar=0, ac=0
    # B_1: br=0, bc=2
    # C^T_1: cr=2, cc=0
    col = add_strassen(U, V, W, col, 0, 0, 0, 2, 2, 0)
    
    # A_1: ar=0, ac=2
    # B_4: br=2, bc=2
    # C^T_1: cr=2, cc=0
    col = add_strassen(U, V, W, col, 0, 2, 2, 2, 2, 0)
    
    # C^T_2 (1x2) = A_0 B_2 + A_1 B_5
    # A_0: ar=0, ac=0
    # B_2: br=0, bc=4
    # C^T_2: cr=4, cc=0
    col = add_trivial_2x2x1(U, V, W, col, 0, 0, 0, 4, 4, 0)
    
    # A_1: ar=0, ac=2
    # B_5: br=2, bc=4
    # C^T_2: cr=4, cc=0
    col = add_trivial_2x2x1(U, V, W, col, 0, 2, 2, 4, 4, 0)
    
    return {"U": U, "V": V, "W": W}

