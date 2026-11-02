"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x4 * 4x4
    # Split A (2x4) into A_L (2x2) and A_R (2x2)
    # Split B (4x4) into B_TL, B_TR, B_BL, B_BR (each 2x2)
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    # Each 2x2 * 2x2 multiplication uses Strassen's algorithm (rank 7)
    # Total rank = 4 * 7 = 28
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Indices for A: i in {0,1}, k in {0,1,2,3}
    # Indices for B: k in {0,1,2,3}, j in {0,1,2,3}
    # Indices for C: i in {0,1}, j in {0,1,2,3}
    
    # Block definitions:
    # A_L: rows 0-1, cols 0-1 of A. Index in U: i*4 + k
    # A_R: rows 0-1, cols 2-3 of A. Index in U: i*4 + (2+k')
    # B_TL: rows 0-1, cols 0-1 of B. Index in V: k*4 + j
    # B_TR: rows 0-1, cols 2-3 of B. Index in V: k*4 + (2+j')
    # B_BL: rows 2-3, cols 0-1 of B. Index in V: (2+k')*4 + j
    # B_BR: rows 2-3, cols 2-3 of B. Index in V: (2+k')*4 + (2+j')
    
    # Strassen factors for 2x2 * 2x2:
    # Inputs: A (2x2), B (2x2)
    # A indices: 0,1,2,3 (row-major)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We have 4 such blocks.
    # Block 0: A_L * B_TL -> contributes to C_L (cols 0-1)
    # Block 1: A_R * B_BL -> contributes to C_L (cols 0-1)
    # Block 2: A_L * B_TR -> contributes to C_R (cols 2-3)
    # Block 3: A_R * B_BR -> contributes to C_R (cols 2-3)
    
    # Note: C_L and C_R are added together in the final W matrix.
    
    # Strassen Factor Vectors for A (2x2)
    # A00=0, A01=1, A10=2, A11=3
    S_A = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Strassen Factor Vectors for B (2x2)
    # B00=0, B01=1, B10=2, B11=3
    S_B = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [0, 1, -1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 1, 0],  # M6
        [0, 1, 0, 1]   # M7
    ]
    
    # Strassen Factor Vectors for C (2x2)
    # C00=0, C01=1, C10=2, C11=3
    S_C = [
        [1, 0, 0, 0],  # M1
        [0, 0, 1, 0],  # M2
        [0, 0, 0, 1],  # M3
        [0, 1, 0, 0],  # M4
        [0, 0, 0, 1],  # M5 (Wait, M5 contributes to C00 and C01? No)
        # Let's re-verify C reconstruction
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So W coefficients for each M_r:
        # M1: C00=1, C11=1
        # M2: C10=1, C11=-1
        # M3: C01=1, C11=1
        # M4: C00=1, C10=1
        # M5: C00=-1, C01=1
        # M6: C11=1
        # M7: C00=1
    ]
    
    # Correct S_C mapping (row is M_r, col is C_idx)
    S_C = [
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11
        [0, 1, 0, 1],  # M3 -> C01, C11
        [1, 0, 1, 0],  # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0]   # M7 -> C00
    ]
    
    # Helper to add Strassen block
    def add_block(u_offset, v_offset, w_offset, block_id):
        # u_offset: index in U for A block (0 or 2 for A_L/A_R? No, A_L is cols 0-1, A_R is cols 2-3)
        # A is 2x4. 
        # A_L: i=0,1; k=0,1. U indices: i*4+k
        # A_R: i=0,1; k=2,3. U indices: i*4+k
        # B is 4x4.
        # B_TL: k=0,1; j=0,1. V indices: k*4+j
        # B_TR: k=0,1; j=2,3. V indices: k*4+j
        # B_BL: k=2,3; j=0,1. V indices: k*4+j
        # B_BR: k=2,3; j=2,3. V indices: k*4+j
        # C is 2x4.
        # C_L: i=0,1; j=0,1. W indices: j*2+i (Trace convention W[k*n+i, r]? 
        # Task says: W[k*n+i, r] where k is col of B (j in my notation), i is row of A.
        # So W index = j*2 + i.
        
        col_start = block_id * 7
        for r in range(7):
            # Update U
            for a_idx in range(4):
                val = S_A[r][a_idx]
                if val != 0:
                    # Map local A index (0..3) to global U index
                    # local 0->0, 1->1, 2->2, 3->3
                    # u_offset is the base index in U for this 2x2 block
                    # A_L: base 0. A_R: base 2.
                    u_idx = u_offset + a_idx
                    U[u_idx][col_start + r] += val
                    
            # Update V
            for b_idx in range(4):
                val = S_B[r][b_idx]
                if val != 0:
                    # Map local B index (0..3) to global V index
                    # v_offset is the base index in V for this 2x2 block
                    v_idx = v_offset + b_idx
                    V[v_idx][col_start + r] += val
                    
            # Update W
            for c_idx in range(4):
                val = S_C[r][c_idx]
                if val != 0:
                    # Map local C index (0..3) to global W index
                    # w_offset is the base index in W for this 2x2 block
                    # C_L: j=0,1. W index = j*2 + i.
                    # local C index 0 (C00) -> j=0, i=0 -> 0
                    # local C index 1 (C01) -> j=1, i=0 -> 2
                    # local C index 2 (C10) -> j=0, i=1 -> 1
                    # local C index 3 (C11) -> j=1, i=1 -> 3
                    # So mapping: 0->0, 1->2, 2->1, 3->3
                    local_to_global = [0, 2, 1, 3]
                    w_idx = w_offset + local_to_global[c_idx]
                    W[w_idx][col_start + r] += val

    # Block 0: A_L * B_TL -> C_L
    # A_L: u_offset 0 (cols 0,1)
    # B_TL: v_offset 0 (rows 0,1; cols 0,1)
    # C_L: w_offset 0 (cols 0,1)
    add_block(0, 0, 0, 0)
    
    # Block 1: A_R * B_BL -> C_L
    # A_R: u_offset 2 (cols 2,3)
    # B_BL: v_offset 8 (rows 2,3; cols 0,1) -> k=2,3 => 2*4=8
    # C_L: w_offset 0
    add_block(2, 8, 0, 1)
    
    # Block 2: A_L * B_TR -> C_R
    # A_L: u_offset 0
    # B_TR: v_offset 2 (rows 0,1; cols 2,3) -> j=2,3 => 2
    # C_R: w_offset 2 (cols 2,3) -> j=2,3 => 2*2=4? 
    # W index = j*2 + i.
    # j=2, i=0 -> 4.
    # So w_offset for C_R is 4.
    add_block(0, 2, 4, 2)
    
    # Block 3: A_R * B_BR -> C_R
    # A_R: u_offset 2
    # B_BR: v_offset 10 (rows 2,3; cols 2,3) -> k=2, j=2 => 2*4+2=10
    # C_R: w_offset 4
    add_block(2, 10, 4, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
