"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # We use a block decomposition:
    # A (4x5) = [A_L (4x4) | A_R (4x1)]
    # B (5x5) = [B_T (4x5); B_B (1x5)]
    # C = A_L B_T + A_R B_B
    
    # Further split B_T (4x5) = [B_TL (4x4) | B_TR (4x1)]
    # C = A_L B_TL + A_L B_TR + A_R B_B
    
    # Part 1: A_L (4x4) * B_TL (4x4) -> C_LL (4x4)
    # Rank 49 using Strassen (7 terms * rank 7 for 2x2)
    
    # Part 2: A_L (4x4) * B_TR (4x1) -> C_LR (4x1)
    # Rank 16 (schoolbook)
    
    # Part 3: A_R (4x1) * B_B (1x5) -> C_RB (4x5)
    # Rank 20 (schoolbook)
    
    total_rank = 49 + 16 + 20
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 Strassen ---
    
    # Rank 7 decomposition for 2x2 * 2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    U_2x2 = np.zeros((4, 7), dtype=int)
    V_2x2 = np.zeros((4, 7), dtype=int)
    W_2x2 = np.zeros((4, 7), dtype=int)
    
    # m1 = (a0+a3)(b0+b3) -> c00(+1), c11(+1)
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
    
    # m2 = (a2+a3)b0 -> c10(+1)
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    V_2x2[0, 1] = 1
    W_2x2[2, 1] = 1
    
    # m3 = a0(b1-b3) -> c01(+1), c11(+1)
    U_2x2[0, 2] = 1
    V_2x2[1, 2] = 1; V_2x2[3, 2] = -1
    W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
    
    # m4 = a3(b2-b0) -> c00(+1), c10(+1)
    U_2x2[3, 3] = 1
    V_2x2[2, 3] = 1; V_2x2[0, 3] = -1
    W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
    
    # m5 = (a0+a1)b3 -> c00(-1), c01(+1)
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    V_2x2[3, 4] = 1
    W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
    
    # m6 = (a2-a0)(b0+b1) -> c11(+1)
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    W_2x2[3, 5] = 1
    
    # m7 = (a1-a3)(b2+b3) -> c00(+1)
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    W_2x2[0, 6] = 1
    
    # Strassen terms for 4x4
    # A blocks: 0:A11(0-1,0-1), 1:A12(0-1,2-3), 2:A21(2-3,0-1), 3:A22(2-3,2-3)
    # B blocks: 0:B11, 1:B12, 2:B21, 3:B22
    # C blocks: 0:C11, 1:C12, 2:C21, 3:C22
    
    # Mapping of Strassen terms to block linear combinations
    # Each term is a list of (block_idx, sign) for A and B
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)] # C11(+1), C22(+1) from M1? 
            # Wait, C11 = M1 + M4 - M5 + M7. C22 = M1 - M2 + M3 + M6.
            # The W matrix for the 2x2 decomposition already handles the C block structure relative to the 2x2 product.
            # We just need to map the 2x2 product indices to the 4x4 indices.
            # The 2x2 product result is a 2x2 matrix.
            # For M1, the result is a 2x2 matrix that contributes to C11 and C22?
            # No, M1 is a 2x2 matrix.
            # In the formula C11 = M1 + ..., M1 is the full 2x2 result of (A11+A22)(B11+B22).
            # So M1 contributes to C11 (top-left 2x2) with +1.
            # Does M1 contribute to C22?
            # C22 = M1 - M2 + M3 + M6. Yes, M1 contributes to C22 (bottom-right 2x2) with +1.
            # So for the 2x2 decomposition of M1, the W matrix entries correspond to the 4 entries of the 2x2 result.
            # We need to place these 4 entries into the correct positions in the global W matrix.
        }
    ]
    
    # Let's define the mapping more explicitly.
    # For each Strassen term i (0..6):
    # 1. Determine the linear combination of A blocks: L_A
    # 2. Determine the linear combination of B blocks: L_B
    # 3. Determine the linear combination of C blocks: L_C (with signs)
    # 4. For each rank-1 term k (0..6) in the 2x2 decomposition:
    #    - The U vector for this term is derived from U_2x2[:, k] applied to L_A.
    #    - The V vector for this term is derived from V_2x2[:, k] applied to L_B.
    #    - The W vector for this term is derived from W_2x2[:, k] applied to L_C.
    
    # Block index to 4x4 index mapping
    # Block 0 (00): rows 0,1; cols 0,1
    # Block 1 (01): rows 0,1; cols 2,3
    # Block 2 (10): rows 2,3; cols 0,1
    # Block 3 (11): rows 2,3; cols 2,3
    
    def block_to_indices(block_idx):
        r0 = (block_idx // 2) * 2
        c0 = (block_idx % 2) * 2
        return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
    
    # Strassen definitions:
    # M1: A: [0, 3], B: [0, 3], C: [0, 3] (C11, C22)
    # M2: A: [2, 3], B: [0], C: [2] (C21) -> Wait, C21 = M2 + M4. M2 contributes to C21.
    #     M2 = (A21+A22)B11.
    #     Result is 2x2. It contributes to C21 (bottom-left).
    #     So C block is 2.
    # M3: A: [0], B: [1, -3], C: [1, 3] (C12, C22)
    #     M3 = A11(B12-B22).
    #     C12 = M3 + M5. C22 = ... + M3.
    #     So C blocks 1 and 3.
    # M4: A: [3], B: [2, -0], C: [0, 2] (C11, C21)
    #     M4 = A22(B21-B11).
    #     C11 = ... + M4. C21 = M2 + M4.
    #     So C blocks 0 and 2.
    # M5: A: [0, 1], B: [3], C: [0, 1] (C11, C12)
    #     M5 = (A11+A12)B22.
    #     C11 = ... - M5. C12 = M3 + M5.
    #     So C blocks 0 and 1.
    # M6: A: [2, -0], B: [0, 1], C: [3] (C22)
    #     M6 = (A21-A11)(B11+B12).
    #     C22 = ... + M6.
    #     So C block 3.
    # M7: A: [1, -3], B: [2, 3], C: [0] (C11)
    #     M7 = (A12-A22)(B21+B22).
    #     C11 = ... + M7.
    #     So C block 0.
    
    strassen_defs = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, 1), (1, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    for s_idx in range(7):
        def_A = strassen_defs[s_idx]['A']
        def_B = strassen_defs[s_idx]['B']
        def_C = strassen_defs[s_idx]['C']
        
        for k in range(7):
            # Construct U vector for this term (size 16)
            U_vec = np.zeros(16, dtype=int)
            for a_block, sign in def_A:
                # U_2x2[:, k] has 4 entries corresponding to 2x2 indices
                # 0:00, 1:01, 2:10, 3:11
                # We need to add sign * U_2x2[local_idx, k] to the global U indices
                # The 2x2 indices map to the block's local coordinates
                # local 0 -> (0,0) in block
                # local 1 -> (0,1) in block
                # local 2 -> (1,0) in block
                # local 3 -> (1,1) in block
                
                block_indices = block_to_indices(a_block)
                for local_idx in range(4):
                    val = U_2x2[local_idx, k]
                    if val != 0:
                        r, c = block_indices[local_idx]
                        global_idx = r * 4 + c # A_L is 4x4
                        U_vec[global_idx] += sign * val
            
            # Construct V vector for this term (size 16)
            V_vec = np.zeros(16, dtype=int)
            for b_block, sign in def_B:
                block_indices = block_to_indices(b_block)
                for local_idx in range(4):
                    val = V_2x2[local_idx, k]
                    if val != 0:
                        r, c = block_indices[local_idx]
                        global_idx = r * 4 + c # B_TL is 4x4
                        V_vec[global_idx] += sign * val
            
            # Construct W vector for this term (size 16)
            W_vec = np.zeros(16, dtype=int)
            for c_block, sign in def_C:
                block_indices = block_to_indices(c_block)
                for local_idx in range(4):
                    val = W_2x2[local_idx, k]
                    if val != 0:
                        r, c = block_indices[local_idx]
                        global_idx = r * 4 + c # C_LL is 4x4
                        W_vec[global_idx] += sign * val
            
            # Map to global indices
            # A_L is A[0:4, 0:4]. Global A index: r * 5 + c
            # B_TL is B[0:4, 0:4]. Global B index: r * 5 + c
            # C_LL is C[0:4, 0:4]. Global C index: r * 4 + c (Wait, C is 4x5. C_LL is C[0:4, 0:4]. Index r*4+c? No, C is p x n = 5 x 4.
            # W index is k*n + i. Here k is row of C, i is col of C.
            # C is 5x4.
            # C_LL corresponds to rows 0..3, cols 0..3.
            # So global W index = r * 4 + c.
            
            U_global_idx = np.arange(16)
            # A_L index (r,c) -> r*4+c. Global A index (r,c) -> r*5+c.
            # Mapping: local (r,c) -> r*4+c. Global (r,c) -> r*5+c.
            # We can just loop.
            
            for local_idx in range(16):
                r = local_idx // 4
                c = local_idx % 4
                U_global = r * 5 + c
                V_global = r * 5 + c
                W_global = r * 4 + c
                
                U[U_global, col] = U_vec[local_idx]
                V[V_global, col] = V_vec[local_idx]
                W[W_global, col] = W_vec[local_idx]
            
            col += 1
    
    # --- Part 2: A_L (4x4) * B_TR (4x1) ---
    # A_L: rows 0..3, cols 0..3.
    # B_TR: rows 0..3, col 4.
    # C_LR: rows 0..3, col 0.
    
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[0 * 4 + i, col] = 1 # C row 0, col i? 
            # C is 5x4. C_LR is C[0:4, 0:1].
            # C index (k, i) where k is row, i is col.
            # Here result is C[i, 0]. So k=i, i=0?
            # C = A B. C[i, k] = sum_j A[i, j] B[j, k].
            # We are computing C[i, 0] for i=0..3.
            # So W index is k*n + i = 0*4 + i = i.
            # Wait, my previous mapping for W was r*4+c where r is row, c is col.
            # Here r=i, c=0. So index is i*4 + 0 = 4i?
            # Let's re-verify W index convention.
            # "W[k*n+i,r]". k is row of C, i is col of C.
            # In Part 1, C_LL is C[0:4, 0:4].
            # local (r, c) -> k=r, i=c. Index r*4+c. Correct.
            # In Part 2, C_LR is C[0:4, 0:1].
            # local (r, c) -> k=r, i=c. c is always 0.
            # So index is r*4 + 0 = 4r.
            
            W[4 * i, col] = 1
            col += 1
    
    # --- Part 3: A_R (4x1) * B_B (1x5) ---
    # A_R: rows 0..3, col 4.
    # B_B: row 4, cols 0..4.
    # C_RB: rows 0..3, cols 0..4.
    
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
