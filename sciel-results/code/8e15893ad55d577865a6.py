import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are multiplying A (n x m) by B (m x p) to get C (n x p).
    # n=4, m=4, p=9.
    
    # Strategy:
    # Split B into B1 (4x4) and B2 (4x5).
    # C = A B1 + A B2.
    # 
    # A B1: 4x4 * 4x4. Use Strassen on 2x2 blocks.
    # Split A into 4 blocks of 2x2. Split B1 into 4 blocks of 2x2.
    # Strassen on the 2x2 block structure of 4x4 matrix multiplication.
    # This requires 7 products of 2x2 matrices.
    # Each 2x2 * 2x2 product has rank 7 (Strassen).
    # Total rank for A B1 is 7 * 7 = 49.
    #
    # A B2: 4x4 * 4x5.
    # Split B2 into B2a (4x4) and B2b (4x1).
    # A B2a: 4x4 * 4x4. Rank 49 (same as above).
    # A B2b: 4x4 * 4x1.
    # This is computing 4 dot products of length 4? No.
    # C_b2 is 4x1. C_b2[i] = sum_j A[i,j] B2b[j,0].
    # This is a linear map from (A, B2b) to C_b2.
    # It can be decomposed into 16 rank-1 terms (schoolbook for 4x4x1).
    # Term for (i, j): A[i,j] * B2b[j,0] * (delta at C[i,0]).
    # Rank = 16.
    #
    # Total Rank = 49 + 49 + 16 = 114.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 rank-1 vectors for 2x2x2 Strassen.
    # Local indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1).
    strassen_2x2_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def add_strassen_4x4(i0, j0, k0, U, V, W, col):
        # Adds rank-49 decomposition for A[i0:i0+4, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        # i0, j0, k0 are offsets in the global matrices.
        # The 4x4 block of A is split into 2x2 blocks:
        # A11 at (i0, j0), A12 at (i0, j0+2)
        # A21 at (i0+2, j0), A22 at (i0+2, j0+2)
        # Similarly for B.
        
        # Strassen Products M1..M7 for 4x4*4x4
        # Each Mk is a 2x2 * 2x2 multiplication.
        # We need to add 7 Strassen decompositions, each scaled appropriately.
        
        # M1 = (A11 + A22) * (B11 + B22)
        # The A-part is (A11 + A22). This is a 2x2 matrix.
        # The B-part is (B11 + B22).
        # The result contributes to C11, C12, C21, C22? No.
        # Wait, the Strassen algorithm for 4x4 uses the same structure.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mk is a product of two 2x2 matrices.
        # We can call a helper that adds the rank-7 decomposition for a 2x2*2x2 product
        # with specific linear combinations of the input blocks.
        
        # However, implementing the linear combinations directly in U and V is complex.
        # It's easier to just expand the 7 terms of the 4x4 Strassen, and for each term,
        # expand the 7 terms of the 2x2 Strassen.
        
        # Let's define the coefficients for the 4x4 Strassen terms.
        # u_coeffs[s] is a 4x4 matrix (or list of 16 coeffs) for A?
        # No, u_s is a vector in the space of the 2x2 blocks.
        # For M1:
        # A_part = A11 + A22.
        # B_part = B11 + B22.
        # C_part = C11 + C22? No.
        # M1 contributes +1 to C11 and +1 to C22.
        
        # Let's iterate through the 7 Strassen levels.
        # For each level s (1..7), we have a coefficient for A11, A12, A21, A22
        # and B11, B12, B21, B22.
        
        strassen_4x4_coeffs = [
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # A: 11,12,21,22; B: 11,12,21,22; C: 11,12,21,22
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # M4
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        # Now, for each of these 7 terms, we decompose the 2x2*2x2 product.
        # The 2x2*2x2 product is itself decomposed into 7 rank-1 terms (Strassen).
        # So we are essentially doing a nested Strassen.
        
        for s in range(7):
            u_a, v_b, w_c = strassen_4x4_coeffs[s]
            # u_a[0] is coeff for A11, u_a[1] for A12, etc.
            
            # We need to add the 7 terms of the 2x2 Strassen for the "inner" multiplication.
            # But the "inner" multiplication is between the linear combos.
            # This is equivalent to:
            # Sum_{inner_t} ( (sum_{block} u_a[block] * u_inner_t[block]) 
            #             * (sum_{block} v_b[block] * v_inner_t[block])
            #             * (sum_{block} w_c[block] * w_inner_t[block]) )
            
            # This is getting complicated to implement with loops.
            # Let's just generate the 49 terms explicitly.
            
            # For each inner term t (0..6):
            for t in range(7):
                u_inner, v_inner, w_inner = strassen_2x2_terms[t]
                
                # Now we have a specific rank-1 term in the full space.
                # The A-part is a linear combination of the 4 blocks of A.
                # The vector for A in this term will be non-zero only in the blocks where
                # u_a[block] != 0 and u_inner is non-zero.
                
                # Actually, it's easier to think of it as:
                # The term corresponds to:
                # U_vector = sum_{block_ab} u_a[ab] * (u_inner mapped to block_ab)
                # V_vector = sum_{block_bc} v_b[bc] * (v_inner mapped to block_bc)
                # W_vector = sum_{block_ca} w_c[ca] * (w_inner mapped to block_ca)
                
                # Let's build U_vector, V_vector, W_vector.
                
                # A indices:
                # Block 11 (top-left): i0:i0+2, j0:j0+2
                # Block 12 (top-right): i0:i0+2, j0+2:j0+4
                # Block 21 (bottom-left): i0+2:i0+4, j0:j0+2
                # Block 22 (bottom-right): i0+2:i0+4, j0+2:j0+4
                
                # u_inner has 4 elements: [00, 01, 10, 11] within the 2x2 block.
                
                def map_a_block(block_idx, offset_i, offset_j, coeff, vec):
                    # block_idx: 0=11, 1=12, 2=21, 3=22
                    # offset_i, offset_j: global offset of this block
                    # coeff: scalar from u_a
                    # vec: u_inner vector
                    if coeff == 0:
                        return
                    indices = [
                        (offset_i + 0) * m + (offset_j + 0),
                        (offset_i + 0) * m + (offset_j + 1),
                        (offset_i + 1) * m + (offset_j + 0),
                        (offset_i + 1) * m + (offset_j + 1)
                    ]
                    for idx, val in zip(indices, vec):
                        if val != 0:
                            U[idx][col] += coeff * val
                            
                def map_b_block(block_idx, offset_j, offset_k, coeff, vec):
                    if coeff == 0:
                        return
                    indices = [
                        (offset_j + 0) * p + (offset_k + 0),
                        (offset_j + 0) * p + (offset_k + 1),
                        (offset_j + 1) * p + (offset_k + 0),
                        (offset_j + 1) * p + (offset_k + 1)
                    ]
                    for idx, val in zip(indices, vec):
                        if val != 0:
                            V[idx][col] += coeff * val
                            
                def map_c_block(block_idx, offset_k, offset_i, coeff, vec):
                    if coeff == 0:
                        return
                    indices = [
                        (offset_k + 0) * n + (offset_i + 0),
                        (offset_k + 0) * n + (offset_i + 1),
                        (offset_k + 1) * n + (offset_i + 0),
                        (offset_k + 1) * n + (offset_i + 1)
                    ]
                    for idx, val in zip(indices, vec):
                        if val != 0:
                            W[idx][col] += coeff * val

                # Apply mappings for A
                # u_a[0] is for A11 (offset i0, j0)
                map_a_block(0, i0, j0, u_a[0], u_inner)
                map_a_block(1, i0, j0+2, u_a[1], u_inner)
                map_a_block(2, i0+2, j0, u_a[2], u_inner)
                map_a_block(3, i0+2, j0+2, u_a[3], u_inner)
                
                # Apply mappings for B
                # v_b[0] is for B11 (offset j0, k0)
                map_b_block(0, j0, k0, v_b[0], v_inner)
                map_b_block(1, j0, k0+2, v_b[1], v_inner)
                map_b_block(2, j0+2, k0, v_b[2], v_inner)
                map_b_block(3, j0+2, k0+2, v_b[3], v_inner)
                
                # Apply mappings for C
                # w_c[0] is for C11 (offset k0, i0)
                map_c_block(0, k0, i0, w_c[0], w_inner)
                map_c_block(1, k0, i0+2, w_c[1], w_inner)
                map_c_block(2, k0+2, i0, w_c[2], w_inner)
                map_c_block(3, k0+2, i0+2, w_c[3], w_inner)
                
                col += 1
        return col

    def add_schoolbook_4x1(i0, j0, k0, U, V, W, col):
        # Adds rank-16 decomposition for A[i0:i0+4, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        # C[i0:i0+4, k0:k0+1]
        for i in range(4):
            for j in range(4):
                U[(i0+i)*m + (j0+j)][col] = 1
                V[(j0+j)*p + k0][col] = 1
                W[k0*n + (i0+i)][col] = 1
                col += 1
        return col

    # 1. A B1 (4x4 * 4x4) -> cols 0..48
    col = add_strassen_4x4(0, 0, 0, U, V, W, col)
    
    # 2. A B2a (4x4 * 4x4) -> cols 49..97
    # B2a is B[:, 4:8]
    col = add_strassen_4x4(0, 0, 4, U, V, W, col)
    
    # 3. A B2b (4x4 * 4x1) -> cols 98..113
    # B2b is B[:, 8:9]
    col = add_schoolbook_4x1(0, 0, 8, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
