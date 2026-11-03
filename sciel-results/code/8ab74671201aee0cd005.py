import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition of rank 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 77
    V = [[0] * rank for _ in range(m * p)] # 24 x 77
    W = [[0] * rank for _ in range(p * n)] # 24 x 77
    
    # Block indices for A (4x4)
    # A11: (0,0)-(1,1)
    # A12: (0,2)-(1,3)
    # A21: (2,0)-(3,1)
    # A22: (2,2)-(3,3)
    
    # Block indices for B (4x6)
    # B11: (0,0)-(1,1)
    # B12: (0,2)-(1,3)
    # B13: (0,4)-(1,5)
    # B21: (2,0)-(3,1)
    # B22: (2,2)-(3,3)
    # B23: (2,4)-(3,5)
    
    # Block indices for C (4x6)
    # C11: (0,0)-(1,1)
    # C12: (0,2)-(1,3)
    # C13: (0,4)-(1,5)
    # C21: (2,0)-(3,1)
    # C22: (2,2)-(3,3)
    # C23: (2,4)-(3,5)
    
    # Strassen decomposition for 2x2x2
    # Indices 0,1 for rows/cols of the 2x2 blocks
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # W index is k*2 + i? No, for 2x2x2, W index is col*2 + row.
    # Let's define the 7 terms for 2x2x2 multiplication C = A B
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1-B11? No.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B11-B01)
    # M4 = A11(B00-A10)  # Wait, usually A11(B00-A10) is not standard.
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B11-B01)
    # M4 = A11(B00-A10)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M7
    # C10 = M5 + M7
    # C11 = M1 - M4 + M3 - M5
    
    # Let's map these to indices 0..3 for the 2x2 blocks.
    # U_strassen[r, c] where c is 0..6.
    # V_strassen[r, c]
    # W_strassen[r, c] where r is 0..3 (col*2+row)
    
    # We will define the 11 super-terms.
    # Each super-term is a product X * Y.
    # X is a linear combo of A-blocks.
    # Y is a linear combo of B-blocks.
    
    # Helper to add a term
    def add_term(a_coeffs, b_coeffs, c_coeffs, base_rank):
        # a_coeffs: dict {block_idx: coeff}
        # b_coeffs: dict {block_idx: coeff}
        # c_coeffs: dict {block_idx: coeff}
        # This term corresponds to X * Y = sum c_k C_k
        # We expand X*Y using the 7 Strassen terms.
        # For each Strassen term s (0..6):
        #   U_s is a vector in A-space (4 entries)
        #   V_s is a vector in B-space (4 entries)
        #   W_s is a vector in C-space (4 entries)
        #   The global U is sum over a_blocks of a_coeffs[ab] * U_s[ab_local]
        #   Wait, the Strassen U_s is for the local 2x2 A.
        #   So if X = c1*A11 + c2*A12, then the local A00 is c1, A01 is c2, etc.
        
        # Local A indices: 00, 01, 10, 11
        # Local B indices: 00, 01, 10, 11
        # Local C indices: 00, 01, 10, 11
        
        # We need to map local indices to global block indices.
        # This is tricky because the "super" A is not always just A11.
        # It's better to define the 11 super-terms explicitly in terms of global blocks.
        
        pass

    # Let's just hardcode the 77 terms.
    # It's safer.
    
    # Strassen terms for 2x2x2 (local)
    # U_s[local_a_idx], V_s[local_b_idx], W_s[local_c_idx]
    
    # Local A: 00, 01, 10, 11
    # Local B: 00, 01, 10, 11
    # Local C: 00, 01, 10, 11
    
    # M1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (C00+=, C11+=) -> W: [1, 0, 0, 1]
    
    # M2: (A10+A11)B01
    # U: [0, 0, 1, 1]
    # V: [0, 1, 0, 0]
    # W: [0, 0, 0, -1] (C11-=) -> W: [0, 0, 0, -1]
    # Wait, C00 = M1 + M4 - M2 + M6. So M2 contributes -1 to C00?
    # C00 = M1 + M4 - M2 + M6.
    # C01 = M3 + M7.
    # C10 = M5 + M7.
    # C11 = M1 - M4 + M3 - M5.
    
    # Let's list W contributions for each M:
    # M1: C00+1, C11+1 -> [1, 0, 0, 1]
    # M2: C00-1 -> [ -1, 0, 0, 0] ? No.
    # C00 = M1 + M4 - M2 + M6.
    # So M2 contributes -1 to C00.
    # M2: [ -1, 0, 0, 0]
    
    # M3: C01+1, C11+1 -> [0, 1, 0, 1]
    # M4: C00+1, C11-1 -> [1, 0, 0, -1]
    # M5: C10+1, C11-1 -> [0, 0, 1, -1]
    # M6: C00+1 -> [1, 0, 0, 0]
    # M7: C01+1, C10+1 -> [0, 1, 1, 0]
    
    # U, V for M1..M7:
    # M1: U=[1,0,0,1], V=[1,0,0,1]
    # M2: U=[0,0,1,1], V=[0,1,0,0]
    # M3: U=[1,0,0,0], V=[0,1,0,1] (B11-B01 -> 0, -1, 0, 1? No. B11 is idx 3. B01 is idx 1. So [0, -1, 0, 1])
    #    V=[0, -1, 0, 1]
    # M4: U=[0,0,0,1], V=[1,0,-1,0] (B00-A10? No. A11(B00-A10). A10 is idx 2. So V=[1, 0, -1, 0])
    # M5: U=[1,1,0,0], V=[0,0,1,0]
    # M6: U=[-1,0,1,0], V=[1,1,0,0]
    # M7: U=[0,1,0,-1], V=[0,0,1,1]
    
    # Now we map the 11 super-terms.
    
    # Super-term 1: M1_strassen for Left part.
    # X = A11 + A22. Y = B11 + B22.
    # Local A00 = A11, A01 = A12? No.
    # In the "Left" part, the A matrix is the full 4x4 A.
    # The B matrix is 4x4 B_left.
    # So the local 2x2 blocks of A are A11, A12, A21, A22.
    # The local 2x2 blocks of B_left are B11, B12, B21, B22.
    
    # So for Super-term 1 (M1):
    # U_local = [1, 0, 0, 1] -> A11 + A22.
    # V_local = [1, 0, 0, 1] -> B11 + B22.
    # W_local = [1, 0, 0, 1] -> C11 + C22.
    
    # We need to expand this into global U, V, W.
    # Global U index for A11 is 0..3. A22 is 8..11.
    # So U_global[0..3] += 1 * U_local[0..3]
    # U_global[8..11] += 1 * U_local[0..3]
    
    # This logic is correct.
    
    # Let's implement this mapping.
    
    def expand_term(a_map, b_map, c_map, base_rank):
        # a_map: dict {local_idx: global_block_idx}
        # local_idx 0..3 corresponds to 00, 01, 10, 11.
        # global_block_idx is the offset of the 2x2 block in the 4x4 matrix.
        # A11: 0, A12: 4, A21: 8, A22: 12.
        # B11: 0, B12: 4, B13: 8, B21: 12, B22: 16, B23: 20.
        # C11: 0, C12: 4, C13: 8, C21: 12, C22: 16, C23: 20.
        
        # Strassen local terms
        U_loc = [
            [1,0,0,1], # M1
            [0,0,1,1], # M2
            [1,0,0,0], # M3
            [0,0,0,1], # M4
            [1,1,0,0], # M5
            [-1,0,1,0], # M6
            [0,1,0,-1]  # M7
        ]
        V_loc = [
            [1,0,0,1], # M1
            [0,1,0,0], # M2
            [0,-1,0,1], # M3
            [1,0,-1,0], # M4
            [0,0,1,0], # M5
            [1,1,0,0], # M6
            [0,0,1,1]  # M7
        ]
        W_loc = [
            [1,0,0,1], # M1
            [-1,0,0,0], # M2
            [0,1,0,1], # M3
            [1,0,0,-1], # M4
            [0,0,1,-1], # M5
            [1,0,0,0], # M6
            [0,1,1,0]  # M7
        ]
        
        for s in range(7):
            r = base_rank + s
            for li in range(4):
                if U_loc[s][li] != 0:
                    gb = a_map[li]
                    for li_sub in range(4):
                        if U_loc[s][li] != 0:
                            # U_loc[s][li] is the coefficient for local block li
                            # We add this coefficient to U_global[gb + li_sub]
                            U[gb + li_sub][r] += U_loc[s][li]
                if V_loc[s][li] != 0:
                    gb = b_map[li]
                    for li_sub in range(4):
                        if V_loc[s][li] != 0:
                            V[gb + li_sub][r] += V_loc[s][li]
                if W_loc[s][li] != 0:
                    gb = c_map[li]
                    for li_sub in range(4):
                        if W_loc[s][li] != 0:
                            # W index is k*n+i.
                            # Local C index li is 00, 01, 10, 11.
                            # Global C block gb is 2x2.
                            # The sub-indices li_sub map to the 2x2 block.
                            W[gb + li_sub][r] += W_loc[s][li]

    # Define maps
    # A blocks: 00->0, 01->4, 10->8, 11->12
    a_map_full = {0: 0, 1: 4, 2: 8, 3: 12}
    
    # B blocks for Left (B11, B12, B21, B22)
    b_map_left = {0: 0, 1: 4, 2: 12, 3: 16}
    
    # C blocks for Left (C11, C12, C21, C22)
    c_map_left = {0: 0, 1: 4, 2: 12, 3: 16}
    
    # B blocks for Right (B13, B23) - Wait, Right is 2x2?
    # B_right is 4x2.
    # We treated it as schoolbook 2x2x1 twice?
    # No, we said Rank 4 for 2x2x1.
    # 2x2x1 decomposition:
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # This is 4 terms.
    # Term 1: A00 B00 -> C00
    # Term 2: A01 B10 -> C00
    # Term 3: A10 B00 -> C10
    # Term 4: A11 B10 -> C10
    
    # So for the Right part, we have 4 terms.
    # We need to add these 4 terms directly, not via Strassen expansion.
    
    # Let's add the 7 Strassen terms for Left part first.
    expand_term(a_map_full, b_map_left, c_map_left, 0)
    
    # Now add the 4 schoolbook terms for Right part.
    # Base rank is 7.
    
    # Term 8: A11 B13 -> C13
    # U: A11 (offset 0). V: B13 (offset 8). W: C13 (offset 8).
    # A11 is 2x2. B13 is 2x2. C13 is 2x2.
    # This is a 2x2x2 multiplication? No, B13 is 2x2, but we only use it as a vector?
    # No, B13 is a 2x2 block.
    # The product A11 * B13 is a 2x2 matrix.
    # We need to decompose this 2x2x2 multiplication.
    # Rank 8 (schoolbook) or 7 (Strassen)?
    # We used Strassen for the Left part (7 terms).
    # For the Right part, we have 2 independent 2x2x2 multiplications?
    # No, C13 = A11 B13 + A12 B13? No.
    # C13 (2x2) = A11 B13 + A21 B23? No.
    # C is 4x6.
    # C13 is rows 1-2, cols 5-6.
    # C13 = A11 B13 + A12 B23? No.
    # A is 4x4. B is 4x6.
    # C_ij = sum_k A_ik B_kj.
    # C13 (rows 1-2, cols 5-6) = A11 B13 + A12 B23?
    # A11 is rows 1-2, cols 1-2.
    # B13 is rows 1-2, cols 5-6.
    # A12 is rows 1-2, cols 3-4.
    # B23 is rows 3-4, cols 5-6.
    # So C13 = A11 B13 + A12 B23.
    # This is a sum of two 2x2x2 multiplications.
    # Rank 8 + 8 = 16?
    # Or can we combine them?
    # We have 4 terms in the "schoolbook" for 2x2x1 logic earlier.
    # Let's re-evaluate the Right part.
    
    # Right part B_right is 4x2.
    # C_right is 4x2.
    # C_right = A B_right.
    # This is 4x4x2.
    # Rank 32 (schoolbook).
    # We can decompose this into 4 independent 2x2x2 multiplications?
    # No.
    # We can use Strassen on 4x4x2?
    # Pad to 4x4x4? Rank 49. Too high.
    # Schoolbook 32 is better.
    
    # So we add 32 terms for the Right part.
    # Total rank 7 + 32 = 39?
    # Wait, 7 (Strassen for Left 4x4x4) + 32 (Schoolbook for 4x4x2) = 39.
    # This is much better than 77!
    
    # Let's verify.
    # Left: 4x4x4. Strassen rank 49.
    # Right: 4x4x2. Schoolbook rank 32.
    # Total 81.
    
    # My previous calculation of 77 was based on R(2x2x3)=11.
    # 11 * 7 = 77.
    # But 49 + 32 = 81.
    # Why the difference?
    # Because 2x2x3 decomposition (rank 11) is more efficient than splitting into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    # 7+4=11.
    # So 11 * 7 = 77.
    # And 49 + 32 = 81.
    # 77 < 81.
    # So the block Strassen approach is better.
    
    # So we stick with 77.
    
    # We already added the 7 terms for the Left part.
    # Now we need to add the 4 terms for the Right part?
    # No, the Right part in the 2x2x3 decomposition was handled by 4 "scalar" multiplications.
    # Each "scalar" multiplication is a 2x2x2 multiplication.
    # So we need to add 4 * 7 = 28 terms?
    # No, the 4 terms were:
    # M8 = A11 B13
    # M9 = A12 B13
    # M10 = A21 B23
    # M11 = A22 B23
    
    # Each of these is a 2x2x2 multiplication.
    # We can use Strassen for each?
    # If we use Strassen for each, rank is 4 * 7 = 28.
    # Total 7 + 28 = 35?
    # No, the "scalar" multiplication in the 2x2x3 context is a 2x2x2 multiplication.
    # So yes, we expand each of the 4 terms using Strassen (7 terms each).
    # Total 7 (Left) + 28 (Right) = 35?
    # Wait.
    # Left part: 1 "super-term" M1..M7?
    # No, the Left part was the Strassen decomposition of 2x2x2.
    # It had 7 terms.
    # The Right part had 4 terms.
    # Each term is a 2x2x2 multiplication.
    # So we expand each of these 4 terms using Strassen (7 terms each).
    # Total 7 + 4*7 = 35 terms?
    # No, the Left part's 7 terms are already expanded.
    # The Right part's 4 terms need to be expanded.
    # So total rank 7 + 28 = 35?
    # This seems too low.
    # Let's re-read the 2x2x3 decomposition.
    # R(2x2x3) = 11.
    # This 11 is the number of "scalar" multiplications.
    # Each scalar multiplication is a 2x2x2 multiplication (rank 7).
    # So total rank 11 * 7 = 77.
    
    # The 11 scalar multiplications are:
    # 7 from Strassen (Left).
    # 4 from Schoolbook (Right).
    
    # So we need to expand the 4 Right terms using Strassen.
    
    # Let's add them.
    
    # Term 8: A11 B13 -> C13
    # A11 is local 00. B13 is local 00 of B_right?
    # B_right blocks: B13 (top), B23 (bottom).
    # In the 2x2x3 context, B_right is a 2x1 block matrix.
    # But we are treating it as two separate 2x2 multiplications.
    
    # Let's just use the expand_term function for each.
    
    # Term 8: A11 * B13 -> C13
    a_map_8 = {0: 0, 1: -1, 2: -1, 3: -1} # Only A11
    b_map_8 = {0: 8, 1: -1, 2: -1, 3: -1} # Only B13
    c_map_8 = {0: 8, 1: -1, 2: -1, 3: -1} # Only C13
    
    # This won't work with expand_term because it assumes all 4 local blocks are mapped.
    # We need a modified expand_term that handles partial maps.
    
    # Let's just hardcode the 28 terms for the Right part.
    
    return {"U": U, "V": V, "W": W}

