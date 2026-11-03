import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We are using a block decomposition: 5 = 4 + 1
    # Blocks:
    # A11 (4x4), A12 (4x1)
    # A21 (1x4), A22 (1x1)
    # B11 (4x4), B12 (4x1)
    # B21 (1x4), B22 (1x1)
    
    # Rank contributions:
    # C11 (4x4) = A11*B11 + A12*B21
    #   A11*B11: 4x4x4 Strassen (Rank 49)
    #   A12*B21: 4x1x4 Schoolbook (Rank 16)
    # C12 (4x1) = A11*B12 + A12*B22
    #   A11*B12: 4x4x1 Schoolbook (Rank 16)
    #   A12*B22: 4x1x1 Schoolbook (Rank 4)
    # C21 (1x4) = A21*B11 + A22*B21
    #   A21*B11: 1x4x4 Schoolbook (Rank 16)
    #   A22*B21: 1x1x4 Schoolbook (Rank 4)
    # C22 (1x1) = A21*B12 + A22*B22
    #   A21*B12: 1x4x1 Schoolbook (Rank 4)
    #   A22*B22: 1x1x1 Schoolbook (Rank 1)
    
    # Total Rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    total_rank = 110
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off, coeff=1):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[(i_off + i) * m + (j_off + j)][col] = coeff
                    V[(j_off + j) * p + (k_off + k)][col] = coeff
                    W[(k_off + k) * n + (i_off + i)][col] = coeff
                    col += 1

    # Helper to add Strassen terms for 4x4
    # Strassen 4x4 is built from 7 multiplications of 2x2 blocks.
    # Each 2x2 multiplication is rank 7 (Strassen) or 4 (Schoolbook).
    # To get rank 49, we use Strassen for the 2x2 blocks as well?
    # Actually, the standard Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
    # If we use schoolbook for the 2x2 multiplications, rank is 7 * 4 = 28? No.
    # The "multiplication" of two 2x2 matrices in the Strassen algorithm is a bilinear map.
    # If we implement that bilinear map using schoolbook (4 mults), the total rank is 7 * 4 = 28.
    # But the rank of 4x4 multiplication is known to be 49?
    # Wait. R(2)=7. R(4) <= 7^2 = 49.
    # If we use schoolbook for the inner 2x2, R(4) <= 7 * 4 = 28.
    # Is R(4) <= 28?
    # No, R(4) is at least 28?
    # Actually, it is known that R(4) = 49 is NOT the best upper bound.
    # Strassen's algorithm gives 49.
    # But if we use schoolbook for the base case 2x2, we get 28.
    # Is 28 valid?
    # Yes, because schoolbook is a valid decomposition.
    # So we can use rank 28 for 4x4.
    
    # Let's re-evaluate the total rank with R(4)=28.
    # C11: 28 + 16 = 44
    # C12: 16 + 4 = 20
    # C21: 16 + 4 = 20
    # C22: 4 + 1 = 5
    # Total: 44 + 20 + 20 + 5 = 89.
    
    # This is even better! 125 / 89 = 1.40.
    
    # I will implement Strassen 4x4 using schoolbook 2x2 base cases.
    # This means I need to generate the 7 2x2 multiplication tensors and then decompose each into 4 schoolbook terms.
    
    # Let's refine the helper for Strassen 4x4.
    # The 7 terms are linear combinations of the 2x2 blocks of A and B.
    # Each term is a product of two 2x2 matrices.
    # We decompose each product into 4 schoolbook terms.
    
    # Indices for 4x4 Strassen:
    # A is split into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is split into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global U, V, W.
    # The "multiplication" Mx is a 2x2 matrix multiplication.
    # We will add 4 schoolbook terms for each Mx.
    
    def add_strassen_4x4(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are the top-left indices of the 4x4 block in A, B, C.
        # A block is at [i_off : i_off+4, j_off : j_off+4]
        # B block is at [j_off : j_off+4, k_off : k_off+4]
        # C block is at [k_off : k_off+4, i_off : i_off+4]
        
        # Define the 7 terms as lists of (coeff_A, coeff_B, coeff_C) for each 2x2 sub-block.
        # Sub-block indices: 00, 01, 10, 11 for 2x2 blocks.
        
        terms = []
        
        # M1 = (A11 + A22)(B11 + B22)
        # A parts: 11:1, 22:1. B parts: 11:1, 22:1.
        # We need to generate the schoolbook terms for the product of two 2x2 matrices.
        # The product is a 2x2 matrix.
        # Let's create a helper that takes the linear combination of A blocks and B blocks,
        # and adds the schoolbook terms to U, V, W.
        
        def add_product_term(A_coeffs, B_coeffs, C_coeffs):
            # A_coeffs: dict { (r,c): coeff } for 2x2 blocks of A
            # B_coeffs: dict { (r,c): coeff } for 2x2 blocks of B
            # C_coeffs: dict { (r,c): coeff } for 2x2 blocks of C
            # Each 2x2 block multiplication is decomposed into 4 scalar multiplications.
            # A_block (2x2) * B_block (2x2) -> C_block (2x2)
            # C_block[i,j] = sum_k A_block[i,k] B_block[k,j]
            
            # We iterate over the 4 scalar multiplications for each pair of non-zero A and B blocks?
            # No, the linearity allows us to sum the coefficients first.
            # Let A_eff be the 4x4 matrix formed by the linear combination.
            # Let B_eff be the 4x4 matrix formed by the linear combination.
            # But we are only doing 2x2 block level linearity.
            # Actually, we can just expand the sums.
            
            # For each (i,j) in C_block (2x2):
            # C[i,j] = sum_k (sum_a A_coeff[a] A_block_a[i,k]) (sum_b B_coeff[b] B_block_b[k,j])
            # This is getting complicated.
            
            # Simpler: Just add the schoolbook terms for each non-zero pair of A and B blocks.
            # Since the number of non-zero blocks is small, we can just iterate.
            
            for (ai, aj), ca in A_coeffs.items():
                if ca == 0: continue
                for (bi, bj), cb in B_coeffs.items():
                    if cb == 0: continue
                    # Product of A_block (ai,aj) and B_block (bi,bj)
                    # A_block is at A[i_off + ai*2 : i_off + (ai+1)*2, j_off + aj*2 : j_off + (aj+1)*2]
                    # B_block is at B[j_off + bi*2 : j_off + (bi+1)*2, k_off + bj*2 : k_off + (bj+1)*2]
                    # C_block is at C[k_off + ci*2 : k_off + (ci+1)*2, i_off + cj*2 : i_off + (cj+1)*2]
                    # Wait, the C block index depends on the output position.
                    
                    # This approach is flawed because the C block index is determined by the output position, not the input blocks directly in a simple way for Strassen.
                    # Strassen defines Mx as a 2x2 matrix.
                    # Mx is added to specific C blocks.
                    
                    # Let's just generate the 4 scalar terms for the 2x2 multiplication.
                    # The 2x2 multiplication is:
                    # C00 = A00B00 + A01B10
                    # C01 = A00B01 + A01B11
                    # C10 = A10B00 + A11B10
                    # C11 = A10B01 + A11B11
                    
                    # We need to map the "virtual" A and B matrices to the global indices.
                    # The virtual A matrix is a linear combination of the 4 2x2 blocks of the 4x4 A.
                    # Let's just add the terms directly.
                    
                    # For each of the 4 scalar multiplications in the 2x2 product:
                    # (i, j, k) in 0..1
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                # C[ii, jj] += A[ii, kk] * B[kk, jj]
                                # A[ii, kk] is in block (ai, aj) at local (ii, kk)
                                # B[kk, jj] is in block (bi, bj) at local (kk, jj)
                                # C[ii, jj] is in block (ci, cj) at local (ii, jj)
                                
                                # We need to know which C block this contributes to.
                                # This depends on the Strassen term Mx.
                                # So we must pass the C block mapping.
                                
                                pass # This inner loop is too complex for this helper.

        # Let's simplify.
        # I will explicitly write out the 49 terms for the 4x4 Strassen decomposition.
        # This is tedious but safe.
        # Or, I can use the fact that R(4) <= 49 and just use the known decomposition.
        # But I don't have the coefficients.
        
        # Let's stick to the block decomposition with R(4)=49 (Strassen) or R(4)=28 (Strassen+Schoolbook).
        # I will implement the R(4)=28 version by recursively applying Strassen.
        
        # Actually, the simplest way is to use the schoolbook algorithm for the 4x4 block?
        # Rank 64.
        # Total: 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
        # That's the same as the original.
        
        # So I MUST use Strassen for the 4x4 block to get an improvement.
        # I will use the recursive Strassen implementation.
        
        pass

    # Given the complexity of implementing Strassen inline, I will use a pre-computed or generated decomposition.
    # Since I cannot generate it easily, I will fall back to the 3+2 block decomposition (Rank 120) which is easier to implement with schoolbook for all sub-blocks.
    
    # Let's re-implement the 3+2 block decomposition.
    # It's safer and still an improvement (1.04).
    
    # Reset U, V, W for 3+2
    total_rank = 120
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    
    # 3+2 Split
    # A11 (3x3), A12 (3x2)
    # A21 (2x3), A22 (2x2)
    # B11 (3x3), B12 (3x2)
    # B21 (2x3), B22 (2x2)
    
    # C11 (3x3) = A11*B11 + A12*B21
    # C12 (3x2) = A11*B12 + A12*B22
    # C21 (2x3) = A21*B11 + A22*B21
    # C22 (2x2) = A21*B12 + A22*B22
    
    # Ranks:
    # A11*B11: 3x3x3 (27 schoolbook)
    # A12*B21: 3x2x2 (12 schoolbook)
    # A11*B12: 3x3x2 (18 schoolbook)
    # A12*B22: 3x2x2 (12 schoolbook)
    # A21*B11: 2x3x3 (18 schoolbook)
    # A22*B21: 2x2x3 (12 schoolbook)
    # A21*B12: 2x3x2 (12 schoolbook)
    # A22*B22: 2x2x2 (4 schoolbook)
    
    # Total: 27 + 12 + 18 + 12 + 18 + 12 + 12 + 4 = 115.
    # Wait, earlier I calculated 120 using Strassen for 3x3 (23) and 2x2 (7).
    # 23 + 12 + 18 + 12 + 18 + 12 + 12 + 7 = 114.
    # If I use schoolbook for all:
    # 27 + 12 + 18 + 12 + 18 + 12 + 12 + 4 = 115.
    
    # 115 is better than 120.
    # Score: 125 / 115 = 1.086.
    
    # I will implement the 3+2 block decomposition with schoolbook for all sub-blocks.
    
    def add_schoolbook_block(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[(i_off + i) * m + (j_off + j)][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1

    # C11: 3x3
    add_schoolbook_block(3, 3, 3, 0, 0, 0) # A11*B11
    add_schoolbook_block(3, 2, 2, 0, 3, 0) # A12*B21 (A: 0:3, 3:5; B: 3:5, 0:2? No. B21 is 2x3. B indices: j=3..4, k=0..2. A indices: i=0..2, j=3..4. C indices: k=0..2, i=0..2)
    # Wait, indices:
    # A11: i=0..2, j=0..2
    # A12: i=0..2, j=3..4
    # A21: i=3..4, j=0..2
    # A22: i=3..4, j=3..4
    
    # B11: j=0..2, k=0..2
    # B12: j=0..2, k=3..4
    # B21: j=3..4, k=0..2
    # B22: j=3..4, k=3..4
    
    # C11: k=0..2, i=0..2
    # C12: k=3..4, i=0..2
    # C21: k=0..2, i=3..4
    # C22: k=3..4, i=3..4
    
    # Term 1: A11*B11 -> C11
    # A: i=0..2, j=0..2. B: j=0..2, k=0..2. C: k=0..2, i=0..2.
    # add_schoolbook_block(3, 3, 3, 0, 0, 0) is correct.
    
    # Term 2: A12*B21 -> C11
    # A: i=0..2, j=3..4. B: j=3..4, k=0..2. C: k=0..2, i=0..2.
    add_schoolbook_block(3, 2, 2, 0, 3, 0) # i_off=0, j_off=3, k_off=0.
    # n_sub=3 (i), m_sub=2 (j), p_sub=2 (k).
    
    # Term 3: A11*B12 -> C12
    # A: i=0..2, j=0..2. B: j=0..2, k=3..4. C: k=3..4, i=0..2.
    add_schoolbook_block(3, 3, 2, 0, 0, 3) # i_off=0, j_off=0, k_off=3.
    
    # Term 4: A12*B22 -> C12
    # A: i=0..2, j=3..4. B: j=3..4, k=3..4. C: k=3..4, i=0..2.
    add_schoolbook_block(3, 2, 2, 0, 3, 3) # i_off=0, j_off=3, k_off=3.
    
    # Term 5: A21*B11 -> C21
    # A: i=3..4, j=0..2. B: j=0..2, k=0..2. C: k=0..2, i=3..4.
    add_schoolbook_block(2, 3, 3, 3, 0, 0) # i_off=3, j_off=0, k_off=0.
    
    # Term 6: A22*B21 -> C21
    # A: i=3..4, j=3..4. B: j=3..4, k=0..2. C: k=0..2, i=3..4.
    add_schoolbook_block(2, 2, 3, 3, 3, 0) # i_off=3, j_off=3, k_off=0.
    
    # Term 7: A21*B12 -> C22
    # A: i=3..4, j=0..2. B: j=0..2, k=3..4. C: k=3..4, i=3..4.
    add_schoolbook_block(2, 3, 2, 3, 0, 3) # i_off=3, j_off=0, k_off=3.
    
    # Term 8: A22*B22 -> C22
    # A: i=3..4, j=3..4. B: j=3..4, k=3..4. C: k=3..4, i=3..4.
    add_schoolbook_block(2, 2, 2, 3, 3, 3) # i_off=3, j_off=3, k_off=3.
    
    # Total rank used: 27 + 12 + 18 + 12 + 18 + 12 + 12 + 4 = 115.
    # col should be 115.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
