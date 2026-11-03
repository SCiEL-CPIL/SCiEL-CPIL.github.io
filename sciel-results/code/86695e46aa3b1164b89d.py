import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x4 matrices (flattened)
    # Actually, U is (n*m, rank) = (4, 7)
    # V is (m*p, rank) = (4, 7)
    # W is (p*n, rank) = (4, 7)
    
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Indices for 2x2 matrices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U matrix (A side):
    # Each column corresponds to a product term.
    # We need to extract the linear combinations of A.
    # U[i, k] is the coefficient of A_i in the k-th product's A-factor.
    
    # Let's define the 7 terms' A-factors and B-factors.
    # Term 1: A1 = A11 + A22, B1 = B11 + B22
    # Term 2: A2 = A21 + A22, B2 = B11
    # Term 3: A3 = A11, B3 = B12 - B22
    # Term 4: A4 = A22, B4 = B21 - B11
    # Term 5: A5 = A11 + A12, B5 = B22
    # Term 6: A6 = A21 - A11, B6 = B11 + B12
    # Term 7: A7 = A12 - A22, B7 = B21 + B22
    
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    # U (4x7):
    U_2x2 = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 1, 0, 0, 1, 0, 1],  # A12
        [0, 0, 1, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    
    # V (4x7):
    V_2x2 = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    
    # W (4x7):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 1]   # C22
    ]
    
    # Now we need to construct the 4x4x8 decomposition.
    # We use the tensor product of T_2x2x2 (Strassen) and T_2x2x4 (standard+Strassen)?
    # Actually, T_4x4x8 = T_2x2x2 (for A) \otimes T_2x2x4 (for B)?
    # No, the dimensions are mixed.
    # A is 4x4, B is 4x8.
    # We can view A as 2x2 blocks of 2x2.
    # B as 2x4 blocks of 2x4? No, B is 4x8.
    # Let's use the Kronecker product structure.
    # T_4x4x8 = T_2x2x2 \otimes T_2x2x4?
    # The first T_2x2x2 handles the block indices of A and B.
    # The second T_2x2x4 handles the internal indices.
    
    # Rank of T_2x2x2 is 7.
    # Rank of T_2x2x4 is 14 (two copies of T_2x2x2).
    # Total rank = 7 * 14 = 98.
    
    # We construct U, V, W of size (16, 98), (32, 98), (32, 98).
    # Wait, U is (n*m, rank) = (16, 98).
    # V is (m*p, rank) = (32, 98).
    # W is (p*n, rank) = (32, 98).
    
    # The Kronecker product of tensors:
    # If T1 = sum u1_i v1_j w1_k e_{ijk} and T2 = sum u2_i v2_j w2_k e_{ijk}
    # Then T1 \otimes T2 = sum (u1_i u2_i') (v1_j v2_j') (w1_k w2_k') e_{...}
    # The indices are combined.
    
    # For T_4x4x8:
    # A indices: (a1, a2) where a1, a2 in {0,1,2,3} for 2x2 blocks?
    # No, A is 4x4. Indices (i, j) with i,j in 0..3.
    # We can map (i, j) to (i_block, i_internal, j_block, j_internal).
    # i_block, j_block in {0,1}. i_internal, j_internal in {0,1}.
    # So A[i, j] corresponds to A_block[i_block, j_block] * A_internal[i_internal, j_internal]?
    # No, A is a single matrix.
    # The tensor T_4x4x8 has indices (i, j, k) with i in 0..3, j in 0..3, k in 0..7.
    # We can decompose i, j, k into block and internal parts.
    # i = 2*i_b + i_i
    # j = 2*j_b + j_j
    # k = 2*k_b + k_k (since p=8, k_b in 0..3, k_k in 0..1)
    
    # The tensor is T_{i_b, j_b, k_b} \otimes T_{i_i, j_j, k_k}?
    # No, the multiplication is C[i, k] = sum_j A[i, j] B[j, k].
    # This is not a simple Kronecker product of two independent multiplications.
    # It is a single multiplication.
    
    # However, we can use the Strassen decomposition for the block structure.
    # The block multiplication of 2x2 blocks uses 7 products.
    # Each product is a 2x2 * 2x4 multiplication.
    # The 2x2 * 2x4 multiplication can be done with 14 products (two Strassen 2x2x2).
    # So total 7 * 14 = 98 products.
    
    # We will construct the U, V, W matrices by iterating over the 7 Strassen terms for the blocks,
    # and for each, the 14 Strassen terms for the internal 2x2x4 multiplication.
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # U indices: i in 0..3, j in 0..3 -> i*4+j
    # V indices: j in 0..3, k in 0..7 -> j*8+k
    # W indices: k in 0..7, i in 0..3 -> k*4+i
    
    # Block Strassen terms (7)
    # Internal 2x2x4 terms (14)
    # Total 98 terms.
    
    col = 0
    for b in range(7):
        # For each block term, we need to perform 2x2 * 2x4 multiplication.
        # The 2x2 * 2x4 multiplication is T_2x2x4.
        # T_2x2x4 = T_2x2x2 (for k=0,1) + T_2x2x2 (for k=2,3)?
        # No, k is 0..7.
        # The internal k is 0..1 for the 2x2x2?
        # Wait, B is 2x4. So internal B is 2x4.
        # The internal multiplication is 2x2 * 2x4.
        # This is two 2x2 * 2x2 multiplications side by side.
        # So we use Strassen twice.
        
        for internal_part in range(2):
            # Each internal_part corresponds to a 2x2x2 Strassen decomposition.
            # The k-index for this part is 2*internal_part + 0 and 2*internal_part + 1?
            # No, B is 2x4. The columns are 0,1,2,3.
            # Part 0 uses columns 0,1. Part 1 uses columns 2,3.
            
            for s in range(7):
                # s is the Strassen term index for the internal 2x2x2.
                
                # Determine the A-block, B-block, C-block for the outer Strassen term b.
                # And the A-internal, B-internal, C-internal for the inner Strassen term s.
                
                # Outer Strassen (b):
                # A_block_factor: U_2x2[:, b]
                # B_block_factor: V_2x2[:, b]
                # C_block_factor: W_2x2[:, b]
                
                # Inner Strassen (s):
                # A_internal_factor: U_2x2[:, s]
                # B_internal_factor: V_2x2[:, s]
                # C_internal_factor: W_2x2[:, s]
                
                # The total A-factor is the Kronecker product of the block and internal factors?
                # A[i, j] = A_block[i_b, j_b] * A_internal[i_i, j_j]?
                # No, A is a single 4x4 matrix.
                # The decomposition sums over block and internal indices.
                # The coefficient for A[i, j] in the product term is:
                # U_block[i_b, b] * U_internal[i_i, s]
                # where i = 2*i_b + i_i, j = 2*j_b + j_j.
                
                # Similarly for B[j, k]:
                # j = 2*j_b + j_j
                # k = 2*k_b + k_k?
                # Wait, B is 4x8.
                # Block j_b, internal j_j.
                # Block k_b? No, k is 0..7.
                # We split k into k_block (0..3) and k_internal (0..1).
                # So k = 2*k_block + k_internal.
                # The internal multiplication is 2x2 * 2x2?
                # No, B_block is 2x4.
                # So B_block[j_b, k_block] is a 2x2 matrix?
                # No, B is 4x8.
                # Block structure: 2x2 blocks of size 2x4.
                # So B_block[j_b, k_b] is a 2x4 matrix.
                # The internal multiplication is 2x2 (A_internal) * 2x4 (B_internal).
                # This is two 2x2x2 multiplications.
                # So for each internal_part (0 or 1), we use a 2x2x2 Strassen decomposition.
                # internal_part 0 uses B columns 0,1 (k_internal 0,1 for k_block 0,1?).
                # No, k_block is 0..3.
                # The 2x4 matrix has columns 0,1,2,3.
                # We split it into two 2x2 matrices: cols 0,1 and cols 2,3.
                # So internal_part 0 uses cols 0,1. internal_part 1 uses cols 2,3.
                # For internal_part 0, the k-values are 2*0+0, 2*0+1? No.
                # The k-index in the full matrix is k.
                # k = 4*j_b + ...? No.
                # B is 4x8. Indices (j, k).
                # j = 2*j_b + j_j.
                # k = 4*k_b + k_jj? No.
                # Let's define the mapping carefully.
                
                # A[i, j]: i=2*i_b+i_i, j=2*j_b+j_j.
                # B[j, k]: j=2*j_b+j_j, k=2*k_b+k_k? No, k is 0..7.
                # Let k = 2*k_b + k_k, where k_b in 0..3, k_k in 0..1.
                # But B_block[j_b, k_b] is 2x2?
                # If B is 4x8, and we use 2x2 blocks, then there are 2x4 blocks.
                # So B_block[j_b, k_b] is a 2x2 matrix.
                # Then the internal multiplication is 2x2 * 2x2.
                # So we use Strassen for the internal 2x2x2.
                # This matches the 14 terms (7 for k_b=0,1? No, k_b=0..3).
                # Wait, if k_b is 0..3, then we have 4 blocks of 2x2.
                # So we need 4 Strassen decompositions?
                # Then total rank = 7 (outer) * 7 (inner) * 4 (k_blocks) = 196.
                # This is worse than 98.
                
                # Let's stick to the 2x4 block structure.
                # B_block[j_b, k_b] is 2x4.
                # Internal multiplication: 2x2 * 2x4.
                # This is two 2x2x2 multiplications.
                # So 2 Strassen decompositions per block.
                # Total 7 * 2 * 7 = 98.
                
                # Mapping:
                # i = 2*i_b + i_i
                # j = 2*j_b + j_j
                # k = 4*k_b + 2*k_int_part + k_k?
                # No, k is 0..7.
                # Let k = 2*k_block + k_internal, where k_block in 0..3, k_internal in 0..1.
                # But B_block[j_b, k_block] is 2x2?
                # If B is 4x8, and we use 2x2 blocks, then k_block is 0..3.
                # Then B_block[j_b, k_block] is 2x2.
                # Then internal is 2x2 * 2x2.
                # So we need 4 Strassen decompositions for the internal part.
                # Total 7 * 7 * 4 = 196.
                
                # To get 98, we must use 2x4 blocks for B.
                # So B_block[j_b, k_b] is 2x4.
                # Then k_b is 0..1.
                # k = 4*k_b + 2*j_j? No.
                # k = 4*k_b + 2*k_jj + k_k?
                # Let's define k = 4*k_b + 2*k_i + k_j?
                # This is getting complicated.
                
                # I will use the 196 rank decomposition if it's easier to code correctly.
                # Or I can just use the schoolbook 128.
                # 128 is better than 196.
                # So I should use the schoolbook decomposition if I can't get 98 easily.
                
                # Let's try to code the 98 rank one carefully.
                # Outer Strassen (7 terms).
                # For each term, we have a 2x2 * 2x4 multiplication.
                # The 2x2 * 2x4 multiplication uses 14 terms.
                # The 14 terms are 7 for the first 2x2 of B, and 7 for the second 2x2 of B.
                
                # B is 2x4. Columns 0,1 and 2,3.
                # Part 0: B[:, 0:2]. Part 1: B[:, 2:4].
                # For Part 0, k-values are 0,1.
                # For Part 1, k-values are 2,3.
                
                # In the full matrix, k is 0..7.
                # k = 4*k_b + 2*k_part + k_k?
                # No, k_b is the block index (0..1).
                # k_part is the part index (0..1).
                # k_k is the internal index (0..1).
                # So k = 4*k_b + 2*k_part + k_k.
                
                # Let's verify:
                # k_b=0, k_part=0, k_k=0 -> k=0
                # k_b=0, k_part=0, k_k=1 -> k=1
                # k_b=0, k_part=1, k_k=0 -> k=2
                # k_b=0, k_part=1, k_k=1 -> k=3
                # k_b=1, k_part=0, k_k=0 -> k=4
                # ...
                # k_b=1, k_part=1, k_k=1 -> k=7
                # This works.
                
                # So for each outer term b, and inner part p (0,1), and inner Strassen term s:
                # A-factor: U_block[i_b, b] * U_internal[i_i, s]
                # B-factor: V_block[j_b, b] * V_internal[j_j, s]
                # C-factor: W_block[i_b, b] * W_internal[i_i, s] ? No.
                # C is 2x4.
                # C_block[i_b, k_b] is 2x4.
                # C_internal[i_i, k] is the value.
                # The C-factor for the term is W_block[i_b, b] * W_internal[i_i, s].
                # And the k-index is determined by k_b and k_part and k_k.
                
                # Let's code this.
                
                U_block = U_2x2[:, b]
                V_block = V_2x2[:, b]
                W_block = W_2x2[:, b]
                
                U_internal = U_2x2[:, s]
                V_internal = V_2x2[:, s]
                W_internal = W_2x2[:, s]
                
                for i_b in range(2):
                    for i_i in range(2):
                        i = 2*i_b + i_i
                        coeff_A = U_block[i_b] * U_internal[i_i]
                        if coeff_A != 0:
                            U[i*4 + 0][col] = coeff_A # j_b=0, j_j=0? No, j is summed over.
                            # Wait, U is (n*m, rank).
                            # U[i, j, col].
                            # We need to fill U[i*4+j][col].
                            # The A-factor is a linear combination of A[i, j].
                            # The coefficient is U_block[i_b, b] * U_internal[i_i, s] * U_internal[j_j, s]?
                            # No, the internal Strassen term s has A-factor U_internal[:, s].
                            # So the A-factor for the term is U_block[i_b, b] * U_internal[i_i, s] for A[i, j]?
                            # No, A is 2x2 in the internal part.
                            # So A_internal[i_i, j_j].
                            # The coefficient is U_block[i_b, b] * U_internal[i_i, s] * U_internal[j_j, s]?
                            # No, U_internal is a vector of length 4.
                            # U_internal[i_i*2+j_j, s].
                            # So the coefficient for A[i, j] is U_block[i_b, b] * U_internal[i_i*2+j_j, s].
                            # But j is also a block index.
                            # j = 2*j_b + j_j.
                            # So the coefficient is U_block[i_b, b] * U_block[j_b, b] * U_internal[i_i*2+j_j, s]?
                            # No, the outer Strassen term b uses A_block[i_b, j_b].
                            # So the A-factor is A_block[i_b, j_b] * A_internal[i_i, j_j].
                            # The coefficient for A[i, j] is U_block[i_b, b] * U_block[j_b, b] * U_internal[i_i*2+j_j, s].
                            
                            # This is correct.
                            
                            for j_b in range(2):
                                for j_j in range(2):
                                    j = 2*j_b + j_j
                                    coeff = U_block[i_b] * U_block[j_b] * U_internal[i_i*2+j_j, s]
                                    if coeff != 0:
                                        U[i*4+j][col] = coeff
                                        
                for j_b in range(2):
                    for j_j in range(2):
                        j = 2*j_b + j_j
                        for k_b in range(2):
                            for k_part in range(2):
                                for k_k in range(2):
                                    k = 4*k_b + 2*k_part + k_k
                                    # B-factor: V_block[j_b, b] * V_internal[j_j, s] * V_internal[k_part*2+k_k, s]?
                                    # No, B is 2x4.
                                    # B_block[j_b, k_b] is 2x4.
                                    # B_internal[j_j, k] is the value.
                                    # The internal Strassen term s uses B_internal[j_j, k_internal].
                                    # k_internal is 0..3.
                                    # k_internal = 2*k_part + k_k.
                                    # So the coefficient is V_block[j_b, b] * V_internal[j_j, s] * V_internal[k_part*2+k_k, s]?
                                    # No, V_internal is a vector of length 4.
                                    # V_internal[j_j*2+k_internal, s].
                                    # So the coefficient is V_block[j_b, b] * V_internal[j_j*2+k_part*2+k_k, s].
                                    
                                    coeff = V_block[j_b] * V_internal[j_j*2 + k_part*2 + k_k, s]
                                    if coeff != 0:
                                        V[j*8+k][col] = coeff
                                        
                for i_b in range(2):
                    for i_i in range(2):
                        i = 2*i_b + i_i
                        for k_b in range(2):
                            for k_part in range(2):
                                for k_k in range(2):
                                    k = 4*k_b + 2*k_part + k_k
                                    # C-factor: W_block[i_b, b] * W_internal[i_i, s]
                                    # C is 2x4.
                                    # C_block[i_b, k_b] is 2x4.
                                    # C_internal[i_i, k_internal] is the value.
                                    # k_internal = 2*k_part + k_k.
                                    # So the coefficient is W_block[i_b, b] * W_internal[i_i*2 + k_part*2 + k_k, s].
                                    
                                    coeff = W_block[i_b] * W_internal[i_i*2 + k_part*2 + k_k, s]
                                    if coeff != 0:
                                        W[k*4+i][col] = coeff
                                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
