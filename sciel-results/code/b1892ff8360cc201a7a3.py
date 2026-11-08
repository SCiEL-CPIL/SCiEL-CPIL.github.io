import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target: A (4x4) * B (4x8) = C (4x8)
    # Decompose B into B_left (4x4) and B_right (4x4)
    # C_left = A * B_left, C_right = A * B_right
    # We use the rank 49 decomposition for 4x4x4 multiplication (recursive Strassen).
    # Total rank = 49 + 49 = 98.
    
    R = 98
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Define the rank 7 decomposition for 2x2x2 (Strassen)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1; S_W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1; S_W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1; S_W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = 1
    
    # Define the 7 Strassen "super-terms" for 4x4x4
    # Each super-term is a product of two 2x2 matrices formed by linear combinations of 2x2 blocks.
    # We define for each super-term t:
    # - A_blocks: list of (block_idx, coeff) where block_idx is (I, J) in 0..1
    # - B_blocks: list of (block_idx, coeff)
    # - C_blocks: list of (block_idx, coeff) for the result
    # The product is sum_A * sum_B, added to C blocks.
    
    # Standard Strassen for 4x4 (blocks 00, 01, 10, 11):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # Wait, let's verify M3 and M4 definitions against standard Strassen.
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # Let's check C00:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00
    # -M5 = -A00B11 - A01B11
    # M6 = A10B00 + A10B01 - A00B00 - A00B01
    # Sum: A00B00 (1-1) + A00B11 (1-1) + A11B00 (1-1) + A11B11 + A11B10 - A01B11 + A10B00 + A10B01 - A00B01
    # This does not look like C00 = A00B00 + A01B10.
    
    # Let's use the correct Strassen formulas.
    # Reference:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # Actually, the standard reconstruction is:
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # Let's re-verify C01 = M4 + M5.
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # Sum = A11B10 - A11B00 + A00B11 + A01B11.
    # This is not C01 = A00B01 + A01B11.
    
    # There are different Strassen variants. Let's use one that is easy to map.
    # Variant from Wikipedia:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # Wait, the Wikipedia formula for C01 is M4 + M5?
    # Let's check C01 = A00B01 + A01B11.
    # My M4 and M5 don't seem to produce this.
    
    # Let's use the "Berkovich" or another standard set.
    # Actually, let's just use the rank 7 decomposition directly for 2x2x2 and apply it recursively.
    # The rank 7 decomposition of the 2x2x2 tensor T is given by vectors u_k, v_k, w_k.
    # For 4x4x4, we view it as 2x2 blocks of 2x2 matrices.
    # The tensor is T_4 = sum_{I,J,K} T_{IJK} \otimes T_2.
    # This is not quite right. T_4 = sum_{i,j,k} T_{ijk} e_i \otimes e_j \otimes e_k.
    # If we group indices into blocks, T_4 = sum_{I,J,K} (sum_{i_l, j_l, k_l} T_{I i_l, J j_l, K k_l} e_{i_l} \otimes e_{j_l} \otimes e_{k_l}) \otimes e_I \otimes e_J \otimes e_K.
    # This is T_4 = sum_{I,J,K} T_{IJK}^{block} \otimes e_I \otimes e_J \otimes e_K.
    # Where T_{IJK}^{block} is the 2x2x2 tensor for the block multiplication.
    # If we use the rank 7 decomposition for T_2, we can substitute.
    # But the "scalars" in the block decomposition are actually 2x2 matrices.
    # So we need to decompose the product of two 2x2 matrices (which is T_2) using rank 7.
    # This is exactly what recursive Strassen does.
    
    # Let's implement the recursive Strassen logic programmatically.
    
    def add_strassen_4x4(U, V, W, offset, B_offset_k):
        # Adds 49 terms to U, V, W starting at column offset.
        # B_offset_k is 0 for left half, 4 for right half.
        
        # Define the 7 super-terms for 4x4.
        # Each term is a product of two linear combinations of blocks.
        # We represent the linear combination by a 2x2 matrix of coefficients for the blocks.
        # A_coef[term][I, J] is the coefficient of block A[I, J].
        # B_coef[term][I, J] is the coefficient of block B[I, J].
        # C_coef[term][I, J] is the coefficient of block C[I, J] in the reconstruction.
        # Note: The product of the linear combinations is added to the C blocks.
        # So for each term t, we compute P_t = L_A^t L_B^t.
        # Then C = sum_t C_coef[t] * P_t.
        # But P_t is a 2x2 matrix of 2x2 matrices? No.
        # L_A^t is a 2x2 matrix. L_B^t is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # Each entry of this 2x2 matrix is a 2x2 matrix product.
        # So we need to decompose each of the 4 entries of the product matrix.
        # Each entry is a product of two 2x2 matrices (linear combinations).
        # Wait, L_A^t is a 2x2 matrix where each entry is a linear combination of A entries?
        # No, L_A^t is a 2x2 matrix of 2x2 blocks.
        # L_A^t[I, J] = sum_{I', J'} A_coef[t][I', J'] A[I', J'].
        # This is a 2x2 matrix of 2x2 matrices.
        # The product L_A^t L_B^t is a 2x2 matrix of 2x2 matrices.
        # (L_A^t L_B^t)[I, J] = sum_K L_A^t[I, K] L_B^t[K, J].
        # Each term L_A^t[I, K] L_B^t[K, J] is a product of two 2x2 matrices.
        # We decompose this product using the rank 7 Strassen decomposition for 2x2.
        
        # So for each super-term t:
        # For each I, J in 0..1:
        #   For each K in 0..1:
        #     Let A_mat = L_A^t[I, K] (2x2 matrix)
        #     Let B_mat = L_B^t[K, J] (2x2 matrix)
        #     We add the rank 7 decomposition of A_mat * B_mat to the C[I, J] block.
        #     The contribution to C[I, J] is C_coef[t][I, J] * (A_mat * B_mat).
        
        # This results in 7 (super-terms) * 4 (I, J) * 2 (K) * 7 (rank 7) = 392 terms?
        # No, the rank of the 4x4x4 tensor is 49.
        # My counting is wrong.
        # The rank 7 decomposition of T_2 is T_2 = sum_{s=1}^7 u_s \otimes v_s \otimes w_s.
        # For the block product, we substitute the 2x2 matrices.
        # The total rank is 7 * 7 = 49.
        # The structure is:
        # For each super-term t (1..7):
        #   For each s (1..7) in the inner Strassen:
        #     We have a term.
        
        # Let's trace the indices.
        # Global A index: i = I*2 + i_l, j = J*2 + j_l.
        # Global B index: j = J*2 + j_l, k = B_offset_k + K*2 + k_l.
        # Global C index: i = I*2 + i_l, k = B_offset_k + K*2 + k_l.
        
        # The term (t, s) contributes to U, V, W.
        # U part:
        # u_s is a vector of length 4 (for 2x2).
        # u_s[i_l * 2 + j_l] is the coefficient for A[i_l, j_l] in the inner product.
        # But A[i_l, j_l] is part of a block.
        # The block index is determined by the super-term t.
        # L_A^t[I, K] = sum_{I', J'} A_coef[t][I', J'] A[I', J'].
        # The inner product uses the entries of L_A^t[I, K].
        # Entry (i_l, j_l) of L_A^t[I, K] is sum_{I', J'} A_coef[t][I', J'] A[I', J'][i_l, j_l].
        # So U[i, j] for i = I*2 + i_l, j = J*2 + j_l?
        # No, I and J here are the block indices in the 4x4 matrix.
        # Let's call the block indices I_A, J_A for A.
        # U[I_A*2 + i_l, J_A*2 + j_l] += A_coef[t][I_A, J_A] * u_s[i_l * 2 + j_l] * C_coef[t][I_C, J_C]?
        # No, C_coef is for the block result.
        
        # This is getting complicated. Let's use a known rank 49 decomposition.
        # Or, simpler: use the rank 56 decomposition (Strassen 4x4 with schoolbook 2x2).
        # Rank 56 is safer and still an improvement over 128.
        # Actually, rank 49 is better.
        
        # Let's try to construct the rank 49 decomposition using the recursive property.
        # T_4 = sum_{I,J,K} T_{IJK} \otimes T_2.
        # T_2 = sum_{s=1}^7 u_s \otimes v_s \otimes w_s.
        # So T_4 = sum_{I,J,K} sum_{s=1}^7 T_{IJK} (u_s \otimes v_s \otimes w_s).
        # This is not a direct substitution because T_{IJK} are scalars in the block algebra?
        # No, T_4 is the tensor for 4x4x4 multiplication.
        # It can be written as sum_{i,j,k} e_i \otimes e_j \otimes e_k T_{ijk}.
        # Grouping into blocks:
        # T_4 = sum_{I,J,K} (sum_{i_l, j_l, k_l} T_{I i_l, J j_l, K k_l} e_{i_l} \otimes e_{j_l} \otimes e_{k_l}) \otimes e_I \otimes e_J \otimes e_K.
        # The inner sum is T_2.
        # So T_4 = sum_{I,J,K} T_2 \otimes e_I \otimes e_J \otimes e_K.
        # This implies that the rank of T_4 is at most rank(T_2) * rank(T_2 * T_2 * T_2)?
        # No, this formula is for the tensor product of spaces.
        # The multiplication tensor for 4x4x4 is not simply T_2 \otimes T_2 \otimes T_2.
        # It is the tensor for multiplication in the algebra M_2(M_2).
        
        # Given the complexity, I will implement the rank 56 solution (Strassen 4x4 with schoolbook 2x2).
        # This is a valid improvement (56 < 128).
        
        pass

    # Let's implement the rank 56 solution.
    # 7 Strassen terms for 4x4.
    # Each term is a product of two 2x2 matrices.
    # We use schoolbook (rank 8) for the 2x2 product.
    # Total rank 7 * 8 = 56.
    
    # We need to map the 2x2 Strassen coefficients to the 4x4 blocks.
    
    # Strassen 4x4 coefficients (for the 7 terms):
    # A_coefs[t][I, J]
    # B_coefs[t][I, J]
    # C_coefs[t][I, J]
    
    A_coefs = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    B_coefs = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01 - B11
        [[0, -1], [1, 0]],# M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 1], [0, 1]]  # M7: B10 + B11
    ]
    
    C_coefs = [
        [[1, 0], [0, 1]], # M1: C00 + C11
        [[0, 0], [1, 1]], # M2: C10 + C11
        [[0, 1], [0, 1]], # M3: C01 + C11
        [[1, 0], [0, 1]], # M4: C00 + C10
        [[-1, 1], [0, 1]],# M5: -C00 + C01 + C11
        [[1, 0], [0, 0]], # M6: C00
        [[0, 0], [0, 1]]  # M7: C11
    ]
    
    # Wait, I need to verify the C_coefs.
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    # So:
    # M1: C00, C11
    # M2: C10, C11
    # M3: C01, C11
    # M4: C00, C10
    # M5: -C00, C01, C11
    # M6: C00
    # M7: C11
    
    C_coefs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [0, 1]], # M4
        [[-1, 1], [0, 1]],# M5
        [[1, 0], [0, 0]], # M6
        [[0, 0], [0, 1]]  # M7
    ]
    
    # Now, for each term t, we compute the product of the 2x2 matrices.
    # L_A^t is a 2x2 matrix of 2x2 blocks.
    # L_B^t is a 2x2 matrix of 2x2 blocks.
    # The product is a 2x2 matrix of 2x2 blocks.
    # Each block is a 2x2 matrix product.
    # We use schoolbook for the 2x2 product.
    
    for t in range(7):
        for I in range(2):
            for J in range(2):
                # This is the (I, J) block of the result.
                # It is sum_K L_A^t[I, K] L_B^t[K, J].
                for K in range(2):
                    # L_A^t[I, K] is a 2x2 matrix.
                    # Entry (i_l, j_l) is sum_{I', J'} A_coefs[t][I', J'] A[I', J'][i_l, j_l].
                    # L_B^t[K, J] is a 2x2 matrix.
                    # Entry (k_l, m_l) is sum_{K', J'} B_coefs[t][K', J'] B[K', J'][k_l, m_l].
                    
                    # We add the schoolbook decomposition of the product of these two 2x2 matrices.
                    # The product entry (i_l, m_l) is sum_{j_l, k_l} L_A[i_l, j_l] L_B[j_l, k_l] ... wait.
                    # Product P[I, J][i_l, m_l] = sum_{k_l} L_A^t[I, K][i_l, k_l] L_B^t[K, J][k_l, m_l].
                    
                    # We iterate over the 8 terms of the schoolbook 2x2x2.
                    for i_l in range(2):
                        for k_l in range(2):
                            for m_l in range(2):
                                # This term corresponds to L_A[i_l, k_l] * L_B[k_l, m_l].
                                # Column index:
                                col = t * 8 + I * 4 + J * 2 + K # 7 * 8 = 56? No.
                                # We have 7 terms * 4 blocks (I, J) * 2 K?
                                # No, the rank is 56.
                                # 7 terms * 8 (schoolbook) = 56.
                                # The 8 terms are for the 2x2 product.
                                # But we have 4 blocks (I, J) and 2 K?
                                # This would be 7 * 4 * 2 * 8 = 448 terms.
                                # This is not rank 56.
                                
                                # The rank 56 decomposition is not simply 7 * 8.
                                # It is 7 terms, each of which is a product of two 2x2 matrices.
                                # The rank of the product of two 2x2 matrices is 8 (schoolbook).
                                # So the total rank is 7 * 8 = 56.
                                # This means that for each term t, we add 8 terms.
                                # These 8 terms correspond to the 8 scalar multiplications in the schoolbook 2x2 product.
                                # But the "scalars" are linear combinations of the 4x4 entries.
                                
                                # So for each t, we have 8 terms.
                                # Term (t, s) where s is 0..7.
                                # s corresponds to (i_l, k_l, m_l) in the schoolbook.
                                # But we also need to sum over I, J, K?
                                # No, the Strassen algorithm computes 7 products of 2x2 matrices.
                                # Each product is a 2x2 matrix.
                                # The entries of this 2x2 matrix are added to the C blocks.
                                # So for each t, we compute the 2x2 matrix P_t = L_A^t L_B^t.
                                # P_t is a 2x2 matrix.
                                # We add P_t[I, J] to C[I, J] with coefficient C_coefs[t][I, J].
                                # P_t[I, J] is a 2x2 matrix.
                                # We decompose P_t[I, J] using schoolbook.
                                # So for each t, I, J, we add 8 terms.
                                # Total terms: 7 * 4 * 8 = 224.
                                # This is not 56.
                                
                                # The rank of the 4x4x4 tensor is 49.
                                # The rank of the 2x2x2 tensor is 7.
                                # The rank of the 4x4x4 tensor is 7 * 7 = 49.
                                # This means that the "scalars" in the Strassen algorithm are 2x2 matrices,
                                # and we use the rank 7 decomposition for the 2x2 multiplication.
                                # So for each of the 7 Strassen terms, we use 7 terms for the inner product.
                                # Total 7 * 7 = 49.
                                
                                # So we need to use the rank 7 decomposition for the inner 2x2 product.
                                # This is what I tried to do before.
                                
                                pass

    # Given the complexity, I will return the schoolbook decomposition.
    rank = 128
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i*4+j][column] = 1
                V[j*8+k][column] = 1
                W[k*4+i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
