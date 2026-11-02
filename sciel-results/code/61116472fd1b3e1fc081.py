import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 5x5 using a 4x4 Strassen block and schoolbook for the rest.
    # Rank 4x4 Strassen is 49.
    # Cross terms:
    # A_44 B_41 (4x4 * 4x1) -> Rank 16
    # A_41 B_11 (4x1 * 1x1) -> Rank 4
    # A_14 B_44 (1x4 * 4x4) -> Rank 16
    # A_11 B_14 (1x1 * 1x4) -> Rank 4
    # A_14 B_41 (1x4 * 4x1) -> Rank 4
    # A_11 B_11 (1x1 * 1x1) -> Rank 1
    # Total Rank = 49 + 16 + 4 + 16 + 4 + 4 + 1 = 94
    
    total_rank = 94
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_term(i, j, k, val=1):
        nonlocal col
        if col >= total_rank:
            raise ValueError("Rank exceeded")
        U[i*m + j, col] += val
        V[j*p + k, col] += val
        W[k*n + i, col] += val
        col += 1
        
    # Helper to add a linear combination of terms
    # This is for the Strassen parts, but we can just expand them manually.
    
    # 1. 4x4 Strassen Block (i, j, k in 0..3)
    # We need the rank 49 decomposition for 4x4.
    # This is the tensor product of two rank 7 decompositions?
    # No, it's the composition.
    
    # Let's generate the 4x4 Strassen decomposition explicitly.
    # Strassen 2x2 (rank 7) factors:
    # U_2x2, V_2x2, W_2x2
    
    # Define 2x2 Strassen factors
    # Indices for 2x2: i, j, k in 0..1
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the tensor factors.
    # For a 2x2 multiplication C = AB:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The tensor decomposition is sum of 7 terms.
    # Each term r corresponds to one M_r.
    # U_r is the linear combination of A entries for the first factor of M_r.
    # V_r is the linear combination of B entries for the second factor of M_r.
    # W_r is the linear combination of C entries for the result.
    
    # M1:
    # A part: A00 + A11 -> U[0,0]=1, U[3,0]=1
    # B part: B00 + B11 -> V[0,0]=1, V[3,0]=1
    # C part: C00 + C11 -> W[0,0]=1, W[3,0]=1  (Wait, W index is k*n+i. For 2x2, k*2+i. C00 is k=0,i=0 -> 0. C11 is k=1,i=1 -> 3.)
    
    # Let's verify W index convention: W[k*n + i].
    # For 2x2:
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    # M1 contributes to C00 and C11.
    # W[0,0] = 1, W[3,0] = 1.
    
    # Let's define the 7 terms for 2x2.
    # Term 0 (M1):
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1 (M2): (A10+A11)B00
    # A10 is i=1,j=0 -> idx 2. A11 is i=1,j=1 -> idx 3.
    # B00 is j=0,k=0 -> idx 0.
    # C10 (k=0,i=1 -> 1) and C11 (k=1,i=1 -> 3).
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = 1
    
    # Term 2 (M3): A00(B01-B11)
    # A00 is idx 0.
    # B01 is j=0,k=1 -> idx 1. B11 is j=1,k=1 -> idx 3.
    # C01 (k=1,i=0 -> 2) and C11 (k=1,i=1 -> 3).
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 3 (M4): A11(B10-B00)
    # A11 is idx 3.
    # B10 is j=1,k=0 -> idx 2. B00 is idx 0.
    # C00 (k=0,i=0 -> 0) and C10 (k=0,i=1 -> 1).
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 4 (M5): (A00+A01)B11
    # A00 idx 0, A01 idx 1.
    # B11 idx 3.
    # C00 (0) and C01 (2).
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1  # C00 has -M5, C01 has +M5
    
    # Term 5 (M6): (A10-A00)(B00+B01)
    # A10 idx 2, A00 idx 0.
    # B00 idx 0, B01 idx 1.
    # C11 (3).
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6 (M7): (A01-A11)(B10+B11)
    # A01 idx 1, A11 idx 3.
    # B10 idx 2, B11 idx 3.
    # C00 (0).
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now we have 2x2 factors.
    # To get 4x4, we treat the 4x4 matrix as 2x2 blocks of 2x2 matrices.
    # A_44 = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The Strassen formulas apply to these blocks.
    # Each "multiplication" of 2x2 blocks is itself a rank 7 operation.
    # So we need to expand each M_r (which is a 2x2 matrix product) into 7 terms.
    
    # M1 = (A00+A11)(B00+B11)
    # This is a 2x2 matrix multiplication of (A00+A11) and (B00+B11).
    # Let X = A00+A11, Y = B00+B11.
    # X is a 2x2 matrix. Y is a 2x2 matrix.
    # The product Z = XY has rank 7 decomposition.
    # We need to express the entries of X and Y in terms of the original 4x4 entries.
    
    # This is getting complex to do manually in code without a helper.
    # I will implement a recursive generator for Strassen factors.
    
    def get_strassen_factors(dim):
        if dim == 2:
            return U2, V2, W2
        if dim == 4:
            # Use 2x2 blocks
            # We need to map the 2x2 block indices to 4x4 indices.
            # Block (bi, bj) corresponds to rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
            # U index for 4x4: i*4+j.
            # V index for 4x4: j*4+k.
            # W index for 4x4: k*4+i.
            
            U4 = np.zeros((16, 49))
            V4 = np.zeros((16, 49))
            W4 = np.zeros((16, 49))
            
            # The 7 M terms for 4x4 are computed using the 7 M terms for 2x2 (block level).
            # Let's call the block-level M terms M_block_r.
            # Each M_block_r is a 2x2 matrix product.
            # We expand each M_block_r using the 2x2 factors (U2, V2, W2).
            
            # We need to define the linear combinations for the block-level A and B.
            # Block A00 is A[0:2, 0:2].
            # Block A11 is A[2:4, 2:4].
            
            # Let's define the mapping from block-level indices to 4x4 indices.
            # Block (bi, bj) element (ii, jj) is at 4x4 index (2*bi + ii, 2*bj + jj).
            
            # We will iterate over the 7 block-level terms (r_block in 0..6).
            # For each r_block, we have U2, V2, W2 factors for the 2x2 block multiplication.
            # We need to "inflate" these factors to the 4x4 level.
            
            # The block-level U2 has shape (4, 7).
            # U2[ii*2+jj, r_inner] corresponds to the element (ii, jj) of the block matrix.
            
            # We need to know which block matrix is being used for U2, V2, W2 in each M_block.
            
            # This requires knowing the Strassen formulas for the blocks.
            # M1_block = (A00 + A11) (B00 + B11)
            # M2_block = (A10 + A11) B00
            # ...
            
            # For each M_block_r, we have:
            # A_matrix = sum of some blocks.
            # B_matrix = sum of some blocks.
            # C_matrix = sum of some blocks (with signs).
            
            # We will hardcode the block-level Strassen structure.
            
            # Block indices: 00, 01, 10, 11.
            # M1: A = 00+11, B = 00+11, C = 00+11
            # M2: A = 10+11, B = 00, C = 10+11
            # M3: A = 00, B = 01-11, C = 01+11
            # M4: A = 11, B = 10-00, C = 00+10
            # M5: A = 00+01, B = 11, C = -00+01
            # M6: A = 10-00, B = 00+01, C = 11
            # M7: A = 01-11, B = 10+11, C = 00
            
            block_defs = [
                # (A_blocks, B_blocks, C_blocks)
                # Each is a list of (block_idx, coeff)
                ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M1: 00+11
                ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, 1) ]), # M2: 10+11, 00, 10+11
                ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]), # M3: 00, 01-11, 01+11
                ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]), # M4: 11, 10-00, 00+10
                ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]), # M5: 00+01, 11, -00+01
                ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]), # M6: 10-00, 00+01, 11
                ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]), # M7: 01-11, 10+11, 00
            ]
            
            col4 = 0
            for r_block in range(7):
                A_defs, B_defs, C_defs = block_defs[r_block]
                
                # For each of the 7 inner terms of the 2x2 block multiplication
                for r_inner in range(7):
                    # u_inner is a 4-vector (for 2x2 block)
                    u_inner = U2[:, r_inner]
                    v_inner = V2[:, r_inner]
                    w_inner = W2[:, r_inner]
                    
                    # Expand u_inner to 4x4 U
                    for ii in range(2):
                        for jj in range(2):
                            val_u = u_inner[ii*2+jj]
                            if val_u != 0:
                                for block_idx, coeff in A_defs:
                                    bi = block_idx // 2
                                    bj = block_idx % 2
                                    # 4x4 index
                                    i = 2*bi + ii
                                    j = 2*bj + jj
                                    U4[i*4+j, col4] += val_u * coeff
                                    
                    # Expand v_inner to 4x4 V
                    for jj in range(2):
                        for kk in range(2):
                            val_v = v_inner[jj*2+kk]
                            if val_v != 0:
                                for block_idx, coeff in B_defs:
                                    bi = block_idx // 2
                                    bj = block_idx % 2
                                    # 4x4 index
                                    j = 2*bi + jj
                                    k = 2*bj + kk
                                    V4[j*4+k, col4] += val_v * coeff
                                    
                    # Expand w_inner to 4x4 W
                    for kk in range(2):
                        for ii in range(2):
                            val_w = w_inner[kk*2+ii]
                            if val_w != 0:
                                for block_idx, coeff in C_defs:
                                    bi = block_idx // 2
                                    bj = block_idx % 2
                                    # 4x4 index
                                    k = 2*bi + kk
                                    i = 2*bj + ii
                                    W4[k*4+i, col4] += val_w * coeff
                                    
                    col4 += 1
                    
            return U4, V4, W4
            
    U4, V4, W4 = get_strassen_factors(4)
    
    # Copy 4x4 factors to 5x5 U, V, W
    # 4x4 indices: i*4+j, j*4+k, k*4+i
    # 5x5 indices: i*5+j, j*5+k, k*5+i
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Map 4x4 col to 5x5 col (same col index)
                # But we need to map the values.
                pass
                
    # Actually, we just copy the non-zero entries.
    # Since the 4x4 decomposition only involves i,j,k < 4, we can just copy the blocks.
    # U4 is 16x49. U is 25x94.
    # U[i*5+j, col] = U4[i*4+j, col] for i,j < 4.
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5+j, r] = U4[i*4+j, r]
                V[j*5+k, r] = V4[j*4+k, r] # Wait, k is not in loop
                # Need to loop k too for V and W?
                # No, V4 is 16x49. V4[j*4+k, r].
                # We can just copy the whole 16x49 block into the 25x94 matrix.
    
    # Correct copying:
    # U[0:16, 0:49] is not a direct block copy because indices are interleaved.
    # U4 index i*4+j. U index i*5+j.
    # For i=0, j=0..3: U4[0..3], U[0, 1, 2, 3]
    # For i=1, j=0..3: U4[4..7], U[5, 6, 7, 8]
    
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5+j, r] = U4[i*4+j, r]
                
        for j in range(4):
            for k in range(4):
                V[j*5+k, r] = V4[j*4+k, r]
                
        for k in range(4):
            for i in range(4):
                W[k*5+i, r] = W4[k*4+i, r]
                
    col = 49
    
    # 2. Cross terms
    # A_44 B_41 (i<4, j<4, k=4)
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4)
            
    # A_41 B_11 (i<4, j=4, k=4)
    for i in range(4):
        add_term(i, 4, 4)
        
    # A_14 B_44 (i=4, j<4, k<4)
    for j in range(4):
        for k in range(4):
            add_term(4, j, k)
            
    # A_11 B_14 (i=4, j=4, k<4)
    for k in range(4):
        add_term(4, 4, k)
        
    # A_14 B_41 (i=4, j<4, k=4)
    for j in range(4):
        add_term(4, j, 4)
        
    # A_11 B_11 (i=4, j=4, k=4)
    add_term(4, 4, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
