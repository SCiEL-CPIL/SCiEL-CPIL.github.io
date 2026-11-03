import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
    # We'll build a decomposition using Strassen for 4x4x4 blocks.
    # Split the 4x9 output into two 4x4 blocks and one 4x1 column.
    # C[:, 0:4] = A * B[:, 0:4]  -> 4x4x4, use Strassen (rank 49)
    # C[:, 4:8] = A * B[:, 4:8]  -> 4x4x4, use Strassen (rank 49)
    # C[:, 8:9] = A * B[:, 8:9]  -> 4x4x1, use schoolbook (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    # Strassen decomposition for 4x4x4:
    # C = A * B where A, B are 4x4.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi involves 4 scalar products (2x2x2 schoolbook for the 2x2 blocks),
    # so total rank for 4x4x4 via Strassen = 7 * 4 = 28? No wait.
    # Actually each Mi is a 2x2x2 tensor contraction, which has rank 4 (schoolbook for 2x2x2).
    # So total rank = 7 * 4 = 28. But wait, that's the bilinear complexity.
    #
    # Actually, let me reconsider. For 4x4x4 using Strassen:
    # We reduce to 7 multiplications of 2x2x2 tensors.
    # Each 2x2x2 tensor has rank 4 (schoolbook: 2*2*2 = 8? No, 2x2x2 matrix mult has rank 7 via Strassen, or 8 via schoolbook).
    #
    # Wait, I'm confusing myself. Let me think about this more carefully.
    #
    # The tensor for 4x4x4 matrix multiplication has rank <= 28 using Strassen (7 Strassen multiplies, each being a 2x2x2 tensor contraction which requires 4 scalar multiplications in schoolbook).
    # Actually no. Each "multiply" in Strassen for 2x2 blocks is a 2x2x2 matrix multiplication, which has bilinear complexity 8 (schoolbook) or 7 (Strassen again, but that's for 2x2x2 -> rank 7).
    #
    # So 4x4x4 via one level of Strassen: 7 * 8 = 56 (using schoolbook for the 2x2x2 subproblems)
    # Or 4x4x4 via two levels of Strassen: 7 * 7 = 49 (using Strassen for the 2x2x2 subproblems)
    #
    # The known optimal rank for 4x4x4 is 49 (from Strassen applied twice).
    #
    # Let me construct the 49-rank decomposition for 4x4x4.
    #
    # For 2x2x2 Strassen (rank 7):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A11+A12-A22)*(B21+B22)  -- wait, let me double check
    #
    # Standard Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition. Rank 7."""
        # Indices: A is 2x2, B is 2x2, C is 2x2
        # U has shape (2*2, 7) = (4, 7)
        # V has shape (2*2, 7) = (4, 7)
        # W has shape (2*2, 7) = (4, 7)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[2, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Now build 4x4x4 using Strassen recursively.
    # 4x4x4 = 7 * (2x2x2) where each 2x2x2 uses Strassen (rank 7)
    # So total rank = 7 * 7 = 49.
    
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 decomposition with rank 49."""
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        U2, V2, W2 = strassen_2x2x2()
        
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # 
        # Strassen multiplies:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Index mapping for 4x4:
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # Similarly for B.
        
        # For each Strassen multiply Mi, we need a 2x2x2 contraction.
        # The 2x2x2 has U2 (4x7), V2 (4x7), W2 (4x7).
        # We need to map the 2x2 indices to the 4x4 indices.
        
        # Let's define the block mappings.
        # A11 indices in 4x4: rows 0,1; cols 0,1 -> U indices: 0,1,4,5
        # A12 indices in 4x4: rows 0,1; cols 2,3 -> U indices: 2,3,6,7
        # A21 indices in 4x4: rows 2,3; cols 0,1 -> U indices: 8,9,12,13
        # A22 indices in 4x4: rows 2,3; cols 2,3 -> U indices: 10,11,14,15
        
        # B11: rows 0,1; cols 0,1 -> V indices: 0,1,4,5
        # B12: rows 0,1; cols 2,3 -> V indices: 2,3,6,7
        # B21: rows 2,3; cols 0,1 -> V indices: 8,9,12,13
        # B22: rows 2,3; cols 2,3 -> V indices: 10,11,14,15
        
        # C11: rows 0,1; cols 0,1 -> W indices: 0,1,4,5
        # C12: rows 0,1; cols 2,3 -> W indices: 2,3,6,7
        # C21: rows 2,3; cols 0,1 -> W indices: 8,9,12,13
        # C22: rows 2,3; cols 2,3 -> W indices: 10,11,14,15
        
        # For each Strassen multiply, we place the 2x2x2 decomposition into the 4x4x4.
        # The 2x2x2 U has indices for a 2x2 A-block. We need to map them.
        
        # Let's define a helper that maps 2x2 local indices to 4x4 global indices.
        # Local 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # For A-block at position (br, bc) where br, bc in {0,1}:
        #   global row = br*2 + local_row
        #   global col = bc*2 + local_col
        #   global U index = (br*2 + local_row)*4 + (bc*2 + local_col)
        
        # Same for V and W.
        
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Mi, we have a coefficient pattern for A-blocks and B-blocks.
        # Then we contract using the 2x2x2 Strassen.
        
        # Let's define the multipliers for each block.
        # For Mi: A_side = dict of (br, bc) -> coeff, B_side = dict of (br, bc) -> coeff
        # Then for each 2x2x2 rank term t (0..6):
        #   For each (a_br, a_bc, a_coeff) in A_side:
        #     For each (b_br, b_bc, b_coeff) in B_side:
        #       The product a_coeff * b_coeff * U2[local_a, t] * V2[local_b, t]
        #       goes into U[global_a, rank_offset + t] with weight a_coeff
        #       and V[global_b, rank_offset + t] with weight b_coeff
        #
        # Wait, that's not quite right. The Strassen multiply Mi = (sum_a ca*Aa) * (sum_b cb*Bb)
        # = sum_{a,b} ca*cb * Aa*Bb
        # Each Aa*Bb is a 2x2x2 matrix multiplication.
        # So Mi = sum_{a,b} ca*cb * (Aa*Bb)
        # And each Aa*Bb can be decomposed using the 2x2x2 Strassen.
        # So Mi uses 7 * (number of a's) * (number of b's) rank terms? No...
        #
        # Actually, Mi is a single 2x2x2 matrix multiplication where:
        # The "A" is the 2x2 matrix sum_a ca*Aa
        # The "B" is the 2x2 matrix sum_b cb*Bb
        # So Mi = (sum_a ca*Aa) * (sum_b cb*Bb)
        # This is a single 2x2x2 tensor contraction, which has rank 7.
        # So each Mi contributes 7 rank terms to the 4x4x4 decomposition.
        # Total: 7 * 7 = 49. Correct.
        
        # So for each Mi, we have a 2x2x2 decomposition.
        # The "effective A" for Mi is sum_a ca*Aa, and "effective B" is sum_b cb*Bb.
        # In the 2x2x2 decomposition, U2[local, t] represents the coefficient of A[local] in the t-th rank term.
        # But here, A[local] in the 2x2x2 corresponds to a specific entry of the effective 2x2 matrix.
        # The effective 2x2 matrix's entry (r, c) = sum over (br,bc) such that (br*2+r, bc*2+c) = (r,c)... 
        # No, this is getting confusing. Let me think differently.
        
        # Alternative approach: Just directly compute the 49-rank decomposition by 
        # composing the two levels of Strassen.
        
        # Level 1: 4x4x4 -> 7 "virtual" 2x2x2 multiplications (M1..M7)
        # Level 2: Each 2x2x2 -> 7 scalar multiplications (using Strassen for 2x2x2)
        
        # So we have 49 scalar multiplications total.
        
        # Let me define the composition more carefully.
        
        # For 4x4x4:
        # C = A*B
        # Strassen level 1:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2x2 matrix multiplication.
        # For example, M1 = A1_combined * B1_combined where:
        # A1_combined = A11 + A22 (a 2x2 matrix)
        # B1_combined = B11 + B22 (a 2x2 matrix)
        
        # Now, A1_combined has entries:
        # (A1_combined)[r,c] = A[2*0+r, 2*0+c] + A[2*1+r, 2*1+c] for r,c in {0,1}
        # = A[r, c] + A[2+r, 2+c]
        
        # Similarly for B.
        
        # Now, M1 = A1_combined * B1_combined is a 2x2x2 matrix mult.
        # Using Strassen for this 2x2x2:
        # Let A1c = A1_combined, B1c = B1_combined.
        # S1 = (A1c[0,0]+A1c[1,1])*(B1c[0,0]+B1c[1,1])
        # S2 = (A1c[1,0]+A1c[1,1])*B1c[0,0]
        # S3 = A1c[0,0]*(B1c[0,1]-B1c[1,1])
        # S4 = A1c[1,1]*(B1c[1,0]-B1c[0,0])
        # S5 = (A1c[0,0]+A1c[0,1])*B1c[1,1]
        # S6 = (A1c[1,0]-A1c[0,0])*(B1c[0,0]+B1c[0,1])
        # S7 = (A1c[0,1]-A1c[1,1])*(B1c[1,0]+B1c[1,1])
        
        # M1[0,0] = S1 + S4 - S5 + S7
        # M1[0,1] = S3 + S5
        # M1[1,0] = S2 + S4
        # M1[1,1] = S1 - S2 + S3 + S6
        
        # Each Si is a scalar product of two linear combinations of A and B entries.
        
        # This is getting very detailed. Let me just build it programmatically.
        
        # I'll create a function that given the 4x4x4 tensor, applies the composition.
        
        # Actually, let me just hardcode the 49-rank decomposition by composing the two Strassen levels.
        
        # This is complex, so let me use a different approach: 
        # I'll build the 4x4x4 decomposition by directly using the Kronecker product 
        # of the 2x2x2 Strassen with itself, but with the appropriate mixing.
        
        # Actually, the cleanest way is:
        # 4x4x4 = (2x2x2 ⊗ 2x2x2) with Strassen on both levels.
        # But that's not quite right either.
        
        # Let me just implement it step by step.
        
        # I'll create the 49-rank decomposition by iterating over all 49 scalar products.
        
        # For each of the 7 Strassen multiplies at level 1, and each of the 7 at level 2,
        # I determine which A and B entries are involved.
        
        # Level 1 Strassen multiplies (for 4x4x4):
        # Mi = (sum_{(br,bc) in Ai_blocks} ca_{(br,bc)} * A_{br,bc}) * (sum_{(br,bc) in Bi_blocks} cb_{(br,bc)} * B_{br,bc})
        
        # where A_{br,bc} is the 2x2 submatrix of A at block position (br, bc).
        
        # Level 2 Strassen (for each 2x2x2):
        # For a 2x2x2 mult X*Y = Z where X, Y, Z are 2x2:
        # T1 = (X00+X11)*(Y00+Y11)
        # T2 = (X10+X11)*Y00
        # T3 = X00*(Y01-Y11)
        # T4 = X11*(Y10-Y00)
        # T5 = (X00+X01)*Y11
        # T6 = (X10-X00)*(Y00+Y01)
        # T7 = (X01-X11)*(Y10+Y11)
        
        # Z00 = T1 + T4 - T5 + T7
        # Z01 = T3 + T5
        # Z10 = T2 + T4
        # Z11 = T1 - T2 + T3 + T6
        
        # For the composition, X is the "effective A" for Mi, which is a linear combination of A-blocks.
        # X[r,c] = sum_{(br,bc) in Ai_blocks} ca_{(br,bc)} * A[2*br+r, 2*bc+c]
        
        # So X[r,c] is a linear combination of 4x4 A entries.
        # Similarly Y[r,c] is a linear combination of 4x4 B entries.
        
        # Each Ti = X_{r1,c1} * Y_{r2,c2} (with coefficients from Strassen)
        # So Ti is a product of two linear combinations of A and B entries.
        
        # In the tensor decomposition, each Ti contributes one rank-1 term:
        # U[global_a, rank_idx] = coeff_a
        # V[global_b, rank_idx] = coeff_b
        # W[global_c, rank_idx] = coeff_c (from the Z formula)
        
        # Let me build this systematically.
        
        # Define the level-1 Strassen for 4x4x4.
        # Blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
        
        # M1: A_side = {(0,0):1, (1,1):1}, B_side = {(0,0):1, (1,1):1}
        # M2: A_side = {(1,0):1, (1,1):1}, B_side = {(0,0):1}
        # M3: A_side = {(0,0):1}, B_side = {(0,1):1, (1,1):-1}
        # M4: A_side = {(1,1):1}, B_side = {(1,0):1, (0,0):-1}
        # M5: A_side = {(0,0):1, (0,1):1}, B_side = {(1,1):1}
        # M6: A_side = {(1,0):1, (0,0):-1}, B_side = {(0,0):1, (0,1):1}
        # M7: A_side = {(0,1):1, (1,1):-1}, B_side = {(1,0):1, (1,1):1}
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mi, we have a 2x2x2 Strassen decomposition.
        
        # Let me define the 2x2x2 Strassen in terms of X and Y entries.
        # X is 2x2, Y is 2x2.
        # T1 = (X00+X11)*(Y00+Y11)
        # T2 = (X10+X11)*Y00
        # T3 = X00*(Y01-Y11)
        # T4 = X11*(Y10-Y00)
        # T5 = (X00+X01)*Y11
        # T6 = (X10-X00)*(Y00+Y01)
        # T7 = (X01-X11)*(Y10+Y11)
        
        # Z00 = T1 + T4 - T5 + T7
        # Z01 = T3 + T5
        # Z10 = T2 + T4
        # Z11 = T1 - T2 + T3 + T6
        
        # For each Ti, it's a product of two linear forms.
        # T1 = (X00+X11)*(Y00+Y11)
        #   X-side: {00:1, 11:1}
        #   Y-side: {00:1, 11:1}
        # T2 = (X10+X11)*Y00
        #   X-side: {10:1, 11:1}
        #   Y-side: {00:1}
        # T3 = X00*(Y01-Y11)
        #   X-side: {00:1}
        #   Y-side: {01:1, 11:-1}
        # T4 = X11*(Y10-Y00)
        #   X-side: {11:1}
        #   Y-side: {10:1, 00:-1}
        # T5 = (X00+X01)*Y11
        #   X-side: {00:1, 01:1}
        #   Y-side: {11:1}
        # T6 = (X10-X00)*(Y00+Y01)
        #   X-side: {10:1, 00:-1}
        #   Y-side: {00:1, 01:1}
        # T7 = (X01-X11)*(Y10+Y11)
        #   X-side: {01:1, 11:-1}
        #   Y-side: {10:1, 11:1}
        
        # Z formulas:
        # Z00: T1:1, T4:1, T5:-1, T7:1
        # Z01: T3:1, T5:1
        # Z10: T2:1, T4:1
        # Z11: T1:1, T2:-1, T3:1, T6:1
        
        # Now, for the 4x4x4 decomposition:
        # Each Mi contributes to specific C-blocks with specific signs.
        # And each Mi's 2x2x2 Strassen has 7 terms.
        
        # Total rank = 7 (level 1) * 7 (level 2) = 49.
        
        # Let me build the U, V, W matrices.
        
        U_4x4 = np.zeros((16, 49))
        V_4x4 = np.zeros((16, 49))
        W_4x4 = np.zeros((16, 49))
        
        # Level 1 definitions
        M_defs = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}),              # M2
            ({(0,0):1}, {(0,1):1, (1,1):-1}),            # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}),            # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}),             # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),   # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),   # M7
        ]
        
        # C-block contributions from Mi
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        C_contrib = {
            (0,0): {0:1, 3:1, 4:-1, 6:1},  # C11: M1, M4, -M5, M7
            (0,1): {2:1, 4:1},              # C12: M3, M5
            (1,0): {1:1, 3:1},              # C21: M2, M4
            (1,1): {0:1, 1:-1, 2:1, 5:1},  # C22: M1, -M2, M3, M6
        }
        
        # Level 2 Strassen terms
        T_defs = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # T1
            ({(1,0):1, (1,1):1}, {(0,0):1}),              # T2
            ({(0,0):1}, {(0,1):1, (1,1):-1}),            # T3
            ({(1,1):1}, {(1,0):1, (0,0):-1}),            # T4
            ({(0,0):1, (0,1):1}, {(1,1):1}),             # T5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),   # T6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),   # T7
        ]
        
        # Z contributions from Ti
        Z_contrib = {
            (0,0): {0:1, 3:1, 4:-1, 6:1},  # Z00
            (0,1): {2:1, 4:1},              # Z01
            (1,0): {1:1, 3:1},              # Z10
            (1,1): {0:1, 1:-1, 2:1, 5:1},  # Z11
        }
        
        # Now build the 49-rank decomposition.
        rank_idx = 0
        for mi, (A_side, B_side) in enumerate(M_defs):
            for ti, (X_side, Y_side) in enumerate(T_defs):
                # This rank term corresponds to the ti-th Strassen term of the mi-th level-1 multiply.
                # The X-side and Y-side are in terms of the 2x2 local indices.
                # X[r,c] = sum_{(br,bc) in A_side} A_coeff[(br,bc)] * A[2*br+r, 2*bc+c]
                # Y[r,c] = sum_{(br,bc) in B_side} B_coeff[(br,bc)] * B[2*br+r, 2*bc+c]
                
                # For this rank term, we need to determine:
                # U[global_a, rank_idx] for each A entry involved
                # V[global_b, rank_idx] for each B entry involved
                # W[global_c, rank_idx] for each C entry involved
                
                # The U coefficient for A[global_r, global_c] is:
                #   sum over (br, bc) in A_side such that 2*br = global_r//2... 
                #   Actually, X[r,c] involves A entries at positions (2*br+r, 2*bc+c) for (br,bc) in A_side.
                #   So for a given global A position (gr, gc), it contributes to X[gr%2, gc%2] 
                #   if (gr//2, gc//2) is in A_side, with coefficient A_side[(gr//2, gc//2)].
                #   Then X[gr%2, gc%2] contributes to Ti with coefficient X_side[(gr%2, gc%2)].
                
                # So U[gr*4+gc, rank_idx] = A_side[(gr//2, gc//2)] * X_side[(gr%2, gc%2)]
                # if (gr//2, gc//2) in A_side and (gr%2, gc%2) in X_side.
                
                # Similarly for V.
                
                # For W: C[global_r, global_c] = sum over (br, bc) in C_contrib... 
                # Actually, C_block(br, bc) = sum over mi C_contrib[(br,bc)][mi] * Mi
                # And Mi[local_r, local_c] = sum over ti Z_contrib[(local_r, local_c)][ti] * Ti
                # So C[2*br+local_r, 2*bc+local_c] = sum over mi,ti C_contrib[(br,bc)][mi] * Z_contrib[(local_r, local_c)][ti] * Ti
                # For our rank term (mi, ti), W[2*br+local_r, 2*bc+local_c, rank_idx] = C_contrib[(br,bc)][mi] * Z_contrib[(local_r, local_c)][ti]
                
                # Fill U
                for br in range(2):
                    for bc in range(2):
                        for lr in range(2):
                            for lc in range(2):
                                gr = 2*br + lr
                                gc = 2*bc + lc
                                if (br, bc) in A_side and (lr, lc) in X_side:
                                    U_4x4[gr*4+gc, rank_idx] += A_side[(br,bc)] * X_side[(lr,lc)]
                
                # Fill V
                for br in range(2):
                    for bc in range(2):
                        for lr in range(2):
                            for lc in range(2):
                                gr = 2*br + lr
                                gc = 2*bc + lc
                                if (br, bc) in B_side and (lr, lc) in Y_side:
                                    V_4x4[gr*4+gc, rank_idx] += B_side[(br,bc)] * Y_side[(lr,lc)]
                
                # Fill W
                for br in range(2):
                    for bc in range(2):
                        for lr in range(2):
                            for lc in range(2):
                                gr = 2*br + lr
                                gc = 2*bc + lc
                                if (br, bc) in C_contrib and mi in C_contrib[(br,bc)] and (lr, lc) in Z_contrib and ti in Z_contrib[(lr,lc)]:
                                    W_4x4[gr*4+gc, rank_idx] += C_contrib[(br,bc)][mi] * Z_contrib[(lr,lc)][ti]
                
                rank_idx += 1
        
        return U_4x4, V_4x4, W_4x4
    
    # Now build the full 4x4x9 decomposition.
    # C[:, 0:4] = A * B[:, 0:4] using Strassen 4x4x4 (rank 49)
    # C[:, 4:8] = A * B[:, 4:8] using Strassen 4x4x4 (rank 49)
    # C[:, 8:9] = A * B[:, 8:9] using schoolbook 4x4x1 (rank 16)
    
    U_str, V_str, W_str = strassen_4x4x4()
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))  # m*p = 4*9 = 36
    W = np.zeros((36, total_rank))  # p*n = 9*4 = 36
    
    # First block: C[:, 0:4] = A * B[:, 0:4]
    # B[:, 0:4] is B[j, 0:4] for j in 0..3
    # U for this block: U_str (16x49)
    # V for this block: V_str maps B[0:4, 0:4] which is V_str[j*4+k] -> V[j*9+k] for k=0..3
    # W for this block: W_str maps C[0:4, 0:4] which is W_str[k*4+i] -> W[k*4+i] for k=0..3
    
    U[:, 0:49] = U_str
    for j in range(4):
        for k in range(4):
            V[j*9+k, 0:49] = V_str[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W_str[k*4+i, :]
    
    # Second block: C[:, 4:8] = A * B[:, 4:8]
    U[:, 49:98] = U_str
    for j in range(4):
        for k in range(4):
            V[j*9+(4+k), 49:98] = V_str[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[(4+k)*4+i, 49:98] = W_str[k*4+i, :]
    
    # Third block: C[:, 8:9] = A * B[:, 8:9] using schoolbook 4x4x1
    # Rank 16: U[i*4+j, 98+l] = 1 if l = i*4+j... 
    # Actually for 4x4x1 schoolbook:
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # U[i*4+j, 98+l] = 1 where l indexes the 16 products
    # V[j*9+8, 98+l] = 1
    # W[8*4+i, 98+l] = 1
    
    for i in range(4):
        for j in range(4):
            l = i*4 + j
            U[i*4+j, 98+l] = 1
            V[j*9+8, 98+l] = 1
            W[8*4+i, 98+l] = 1
    
    # Verify
    U_r = U.reshape(4, 4, total_rank)
    V_r = V.reshape(4, 9, total_rank)
    W_r = W.reshape(9, 4, total_rank)
    
    S = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = 144
        U_fb = np.zeros((16, rank_sb))
        V_fb = np.zeros((36, rank_sb))
        W_fb = np.zeros((36, rank_sb))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*9+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
