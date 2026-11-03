import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm applied to 2x2 blocks of 2x2 matrices
    
    rank = 49
    # U is (n*m, rank) = (16, 49)
    # V is (m*p, rank) = (16, 49)
    # W is (p*n, rank) = (16, 49)
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # We need to map the Strassen steps to the scalar indices.
    # Level 1: 7 steps for 4x4
    # Level 2: 7 steps for each 2x2 block product
    
    # Let's define the linear combinations for the 2x2 Strassen step.
    # Input X (2x2), Y (2x2). Output Z (2x2).
    # X indices: 00, 01, 10, 11
    # Y indices: 00, 01, 10, 11
    
    # The 7 multiplications s1..s7 for a 2x2 product:
    # s1 = (x00 + x11)(y00 + y11)
    # s2 = (x10 + x11)y00
    # s3 = x00(y01 - y11)
    # s4 = x11(y10 - y00)
    # s5 = (x00 + x01)y11
    # s6 = (x10 - x00)(y00 + y01)
    # s7 = (x01 - x11)(y10 + y11)
    
    # Result Z:
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    
    # For each of the 7 block products M_l in the 4x4 Strassen step,
    # we define the block-level linear combinations for P_l and Q_l.
    # P_l is a linear combination of A blocks A00, A01, A10, A11.
    # Q_l is a linear combination of B blocks B00, B01, B10, B11.
    
    # A blocks:
    # A00 = A[0:2, 0:2]
    # A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2]
    # A11 = A[2:4, 2:4]
    
    # B blocks:
    # B00 = B[0:2, 0:2]
    # B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2]
    # B11 = B[2:4, 2:4]
    
    # The 7 block products M_l:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We represent the coefficients for A blocks and B blocks for each M_l.
    # coeff_A[l][block_idx] and coeff_B[l][block_idx]
    # block_idx: 0->00, 1->01, 2->10, 3->11
    
    coeff_A = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    coeff_B = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # For each scalar multiplication in the 2x2 Strassen step,
    # we need to know which elements of X (which is P_l) and Y (which is Q_l) are involved.
    # X elements are linear combinations of A blocks.
    # Y elements are linear combinations of B blocks.
    
    # Let's define the 2x2 Strassen terms (x_part, y_part, z_coeff)
    # x_part: indices of X elements involved (0:00, 1:01, 2:10, 3:11) and their coefficients
    # y_part: indices of Y elements involved and their coefficients
    
    strassen_2x2_terms = [
        # s1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # s2
        ([(2, 1), (3, 1)], [(0, 1)]),
        # s3
        ([(0, 1)], [(1, 1), (3, -1)]),
        # s4
        ([(3, 1)], [(2, 1), (0, -1)]),
        # s5
        ([(0, 1), (1, 1)], [(3, 1)]),
        # s6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # s7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    # The contribution of each s_k to Z elements (z00, z01, z10, z11)
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    z_contrib = [
        [1, 0, 0, 1], # s1 -> z00, z11
        [0, 0, 1, -1],# s2 -> z10, z11
        [0, 1, 0, 1], # s3 -> z01, z11
        [1, 0, 1, 0], # s4 -> z00, z10
        [-1, 1, 0, 0],# s5 -> z00, z01
        [0, 0, 0, 1], # s6 -> z11
        [1, 0, 0, 0]  # s7 -> z00
    ]
    # Note: The order in z_contrib is [s1, s2, s3, s4, s5, s6, s7]
    # And the columns are [z00, z01, z10, z11]
    
    # Now we construct U, V, W
    col = 0
    for l in range(7):
        # For each block product M_l
        for s in range(7):
            # For each scalar multiplication s in the 2x2 Strassen step
            
            # Determine the linear combination of A scalars for U
            # U will have non-zero entries for A elements that participate in this multiplication.
            
            # x_part is a list of (x_block_idx, coeff)
            x_part, y_part = strassen_2x2_terms[s]
            
            # For each (x_idx, x_coeff) in x_part:
            #   The element X[x_idx] is a linear combination of A blocks.
            #   X[x_idx] = sum_{b} coeff_A[l][b] * A_block[b]
            #   So the coefficient for A_block[b] in this scalar multiplication is x_coeff * coeff_A[l][b]
            
            # We need to add these coefficients to U for the specific scalar indices.
            
            for x_idx, x_coeff in x_part:
                # x_idx is 0,1,2,3 corresponding to 00, 01, 10, 11 in the 2x2 block P_l
                # The actual A block index is x_idx.
                # The coefficient for this A block in P_l is coeff_A[l][x_idx]
                total_coeff = x_coeff * coeff_A[l][x_idx]
                if total_coeff != 0:
                    # Add total_coeff to U for all scalar indices in A_block[x_idx]
                    # A_block[x_idx] consists of 4 scalars.
                    # Block 00: A[0,0], A[0,1], A[1,0], A[1,1] -> Indices 0, 1, 4, 5
                    # Block 01: A[0,2], A[0,3], A[1,2], A[1,3] -> Indices 2, 3, 6, 7
                    # Block 10: A[2,0], A[2,1], A[3,0], A[3,1] -> Indices 8, 9, 12, 13
                    # Block 11: A[2,2], A[2,3], A[3,2], A[3,3] -> Indices 10, 11, 14, 15
                    
                    # Helper to get indices for a block
                    def get_block_indices(block_idx):
                        r = block_idx // 2
                        c = block_idx % 2
                        base = r * 8 + c * 2
                        return [base, base + 1, base + 4, base + 5]
                    
                    indices = get_block_indices(x_idx)
                    for idx in indices:
                        U[idx, col] += total_coeff
            
            # Similarly for V
            for y_idx, y_coeff in y_part:
                total_coeff = y_coeff * coeff_B[l][y_idx]
                if total_coeff != 0:
                    def get_block_indices_B(block_idx):
                        # B blocks are same structure
                        r = block_idx // 2
                        c = block_idx % 2
                        base = r * 8 + c * 2
                        return [base, base + 1, base + 4, base + 5]
                    
                    indices = get_block_indices_B(y_idx)
                    for idx in indices:
                        V[idx, col] += total_coeff
            
            # For W, we need to distribute the result of this scalar multiplication to the C elements.
            # The scalar multiplication result contributes to Z elements of M_l.
            # Z elements of M_l are blocks of C.
            # M_l is a 2x2 block matrix.
            # The result of the s-th Strassen step for M_l contributes to z00, z01, z10, z11 of M_l.
            # Let's call these local blocks Z_00, Z_01, Z_10, Z_11 of M_l.
            
            # The contribution of s to these local blocks is given by z_contrib[s]
            # z_contrib[s] is a vector [c00, c01, c10, c11]
            
            c_vec = z_contrib[s]
            
            # Now, how do these local blocks of M_l map to the global C blocks?
            # C = sum of M_l blocks with coefficients.
            # C_00 (global block 00) = M1_00 + M4_00 - M5_00 + M7_00
            # C_01 (global block 01) = M3_01 + M5_01
            # C_10 (global block 10) = M2_10 + M4_10
            # C_11 (global block 11) = M1_11 - M2_11 + M3_11 + M6_11
            
            # Let's define the mapping from M_l local block (r, c) to global C block (R, C).
            # M_l is a 2x2 matrix of 2x2 blocks.
            # M_l[r, c] is a 2x2 block.
            
            # Global C block (R, C) where R, C in {0, 1}.
            # C[R, C] is a 2x2 block of 2x2 blocks? No, C is 4x4, so it has 4 blocks of 2x2.
            
            # Mapping:
            # M1 (l=0):
            #   M1_00 -> C_00 (coeff 1)
            #   M1_01 -> C_01 (coeff 0) -- Wait, M1_01 contributes to C_01?
            #   Let's check the C formulas again.
            #   C_00 = M1_00 + M4_00 - M5_00 + M7_00
            #   C_01 = M3_01 + M5_01
            #   C_10 = M2_10 + M4_10
            #   C_11 = M1_11 - M2_11 + M3_11 + M6_11
            
            # So:
            # M1_00 -> C_00 (1)
            # M1_11 -> C_11 (1)
            # M1_01, M1_10 -> 0?
            # Wait, the Strassen formulas for C blocks are:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # This implies that the *entire* block M1 is added to C11?
            # No, M1 is a 2x2 matrix. C11 is a 2x2 matrix.
            # So M1_00 adds to C11_00, M1_01 adds to C11_01, etc.
            
            # Let's verify.
            # C = AB.
            # C11 = A11 B11 + A12 B21.
            # M1 = (A11+A22)(B11+B22) = A11 B11 + A11 B22 + A22 B11 + A22 B22.
            # M4 = A22(B21-B11) = A22 B21 - A22 B11.
            # M5 = (A11+A12)B22 = A11 B22 + A12 B22.
            # M7 = (A12-A22)(B21+B22) = A12 B21 + A12 B22 - A22 B21 - A22 B22.
            
            # Sum M1+M4-M5+M7:
            # A11 B11 + A11 B22 + A22 B11 + A22 B22
            # + A22 B21 - A22 B11
            # - A11 B22 - A12 B22
            # + A12 B21 + A12 B22 - A22 B21 - A22 B22
            # = A11 B11 + (A11 B22 - A11 B22) + (A22 B11 - A22 B11) + (A22 B22 - A22 B22)
            # + (A22 B21 - A22 B21) + (-A12 B22 + A12 B22) + A12 B21
            # = A11 B11 + A12 B21.
            # This is exactly C11.
            
            # So yes, the entire block M1 is added to C11.
            # This means M1_00 adds to C11_00, M1_01 adds to C11_01, M1_10 adds to C11_10, M1_11 adds to C11_11.
            
            # So the mapping is:
            # M_l local block (r, c) maps to Global C block (R, C) with coefficient K.
            # If the global block is (R, C), then the local block (r, c) of M_l contributes to C[R, C][r, c]?
            # No.
            # C11 is a 2x2 block. M1 is a 2x2 block.
            # C11 = M1 + ...
            # So C11[i, j] = M1[i, j] + ...
            # Thus, the element M1[i, j] contributes to C11[i, j].
            # C11 is the top-left 2x2 block of C.
            # So M1[i, j] contributes to C[2*0+i, 2*0+j].
            
            # General Rule:
            # If M_l is added to global block C[R, C] with coefficient K,
            # then M_l[i, j] contributes to C[2*R+i, 2*C+j] with coefficient K.
            
            # Let's define the mapping of M_l to Global Blocks.
            # M1 (l=0):
            #   Adds to C00 (R=0, C=0) with coeff 1.
            #   Adds to C11 (R=1, C=1) with coeff 1.
            #   Wait, C22 = M1 - M2 + M3 + M6.
            #   So M1 adds to C22 (R=1, C=1) with coeff 1?
            #   Let's check indices.
            #   C11 is R=0, C=0.
            #   C12 is R=0, C=1.
            #   C21 is R=1, C=0.
            #   C22 is R=1, C=1.
            
            # M1 adds to C11 (1) and C22 (1).
            # M2 adds to C21 (1) and C22 (-1).
            # M3 adds to C12 (1) and C22 (1).
            # M4 adds to C11 (1) and C21 (1).
            # M5 adds to C11 (-1) and C12 (1).
            # M6 adds to C22 (1).
            # M7 adds to C11 (1).
            
            # Let's verify M5.
            # C11 = M1 + M4 - M5 + M7. So M5 adds to C11 with -1.
            # C12 = M3 + M5. So M5 adds to C12 with 1.
            
            # So for each l, we have a list of (R, C, K) tuples.
            
            mappings = [
                [(0, 0, 1), (1, 1, 1)], # M1
                [(1, 0, 1), (1, 1, -1)],# M2
                [(0, 1, 1), (1, 1, 1)], # M3
                [(0, 0, 1), (1, 0, 1)], # M4
                [(0, 0, -1), (0, 1, 1)],# M5
                [(1, 1, 1)],            # M6
                [(0, 0, 1)]             # M7
            ]
            
            for R, C, K in mappings[l]:
                # The local block Z (result of M_l) has elements z00, z01, z10, z11.
                # These correspond to i=0,1 and j=0,1 in the local 2x2 block.
                # The global C indices are:
                # Row = 2*R + i
                # Col = 2*C + j
                # The coefficient for C[Row, Col] is K * c_vec[z_idx]
                # where z_idx is 0 for (0,0), 1 for (0,1), 2 for (1,0), 3 for (1,1).
                
                # c_vec is [c00, c01, c10, c11]
                # z_idx 0: i=0, j=0
                # z_idx 1: i=0, j=1
                # z_idx 2: i=1, j=0
                # z_idx 3: i=1, j=1
                
                for z_idx in range(4):
                    c_coeff = c_vec[z_idx]
                    if c_coeff == 0:
                        continue
                    final_coeff = K * c_coeff
                    if final_coeff == 0:
                        continue
                    
                    i = z_idx // 2
                    j = z_idx % 2
                    
                    global_i = 2 * R + i
                    global_j = 2 * C + j
                    
                    # C index in flat vector: 4 * global_i + global_j
                    c_idx = 4 * global_i + global_j
                    W[c_idx, col] += final_coeff
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
