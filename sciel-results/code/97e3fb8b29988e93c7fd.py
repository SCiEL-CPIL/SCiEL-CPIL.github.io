import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 1. Define 2x2 Strassen factors
    # Indices for 2x2: A is 2x2 (4 elems), B is 2x2 (4 elems), C is 2x2 (4 elems)
    # C = A * B
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas for 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A indices: 0:A11, 1:A12, 2:A21, 3:A22
    # B indices: 0:B11, 1:B12, 2:B21, 3:B22
    # C indices: 0:C11, 1:C12, 2:C21, 3:C22
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C11, C22 get M1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C21 gets M2
    W2[3, 1] = -1 # C22 gets -M2
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C12 gets M3
    W2[3, 2] = 1 # C22 gets M3
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C11 gets M4
    W2[2, 3] = 1 # C21 gets M4
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C11 gets -M5
    W2[1, 4] = 1 # C12 gets M5
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C22 gets M6
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11 gets M7
    
    # 2. Define Outer Strassen Steps for 4x4
    # A blocks: A00 (0-1,0-1), A01 (0-1,2-3), A10 (2-3,0-1), A11 (2-3,2-3)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # For each of the 7 outer terms, define:
    # - Which A blocks form P_t (and with what signs)
    # - Which B blocks form Q_t (and with what signs)
    # - Which C blocks M_t contributes to (and with what signs)
    
    # P_t = sum(alpha * A_block)
    # Q_t = sum(beta * B_block)
    # M_t contributes to C_block with sign gamma
    
    # Term 1: (A00 + A11)(B00 + B11)
    # P1 = A00 + A11
    # Q1 = B00 + B11
    # M1 contributes to C00 (1), C11 (1)
    
    # Term 2: (A10 + A11)B00
    # P2 = A10 + A11
    # Q2 = B00
    # M2 contributes to C10 (1), C11 (-1)
    
    # Term 3: A00(B01 - B11)
    # P3 = A00
    # Q3 = B01 - B11
    # M3 contributes to C01 (1), C11 (1)
    
    # Term 4: A11(B10 - B00)
    # P4 = A11
    # Q4 = B10 - B00
    # M4 contributes to C00 (1), C10 (1)
    
    # Term 5: (A00 + A01)B11
    # P5 = A00 + A01
    # Q5 = B11
    # M5 contributes to C00 (-1), C01 (1)
    
    # Term 6: (A10 - A00)(B00 + B01)
    # P6 = A10 - A00
    # Q6 = B00 + B01
    # M6 contributes to C11 (1)
    
    # Term 7: (A01 - A11)(B10 + B11)
    # P7 = A01 - A11
    # Q7 = B10 + B11
    # M7 contributes to C00 (1)
    
    outer_terms = [
        # (P_blocks, Q_blocks, C_contribs)
        # P_blocks: list of (block_idx, coeff)
        # block_idx: 0:A00, 1:A01, 2:A10, 3:A11
        # C_contribs: list of (block_idx, coeff)
        [
            [(0, 1), (3, 1)], # P1
            [(0, 1), (3, 1)], # Q1
            [(0, 1), (3, 1)]  # C00, C11
        ],
        [
            [(2, 1), (3, 1)], # P2
            [(0, 1)],         # Q2
            [(2, 1), (3, -1)] # C10, C11
        ],
        [
            [(0, 1)],         # P3
            [(1, 1), (3, -1)],# Q3
            [(1, 1), (3, 1)]  # C01, C11
        ],
        [
            [(3, 1)],         # P4
            [(2, 1), (0, -1)],# Q4
            [(0, 1), (2, 1)]  # C00, C10
        ],
        [
            [(0, 1), (1, 1)], # P5
            [(3, 1)],         # Q5
            [(0, -1), (1, 1)] # C00, C01
        ],
        [
            [(2, 1), (0, -1)],# P6
            [(0, 1), (1, 1)], # Q6
            [(3, 1)]          # C11
        ],
        [
            [(1, 1), (3, -1)],# P7
            [(2, 1), (3, 1)], # Q7
            [(0, 1)]          # C00
        ]
    ]
    
    # Helper to map block index and local 2x2 index to global 4x4 index
    # Block 0 (A00): rows 0-1, cols 0-1
    # Block 1 (A01): rows 0-1, cols 2-3
    # Block 2 (A10): rows 2-3, cols 0-1
    # Block 3 (A11): rows 2-3, cols 2-3
    
    def block_to_global(block_idx, local_idx):
        # local_idx is 0..3 for 2x2
        r_local = local_idx // 2
        c_local = local_idx % 2
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        r_global = r_start + r_local
        c_global = c_start + c_local
        return r_global * 4 + c_global # Flatten row-major for A/B (n*m)
        
    def block_to_global_C(block_idx, local_idx):
        # C is p*n in W? 
        # Prompt: W[k*n+i, r]. 
        # C is n x p. 
        # Block 0 (C00): rows 0-1, cols 0-1
        # Block 1 (C01): rows 0-1, cols 2-3
        # Block 2 (C10): rows 2-3, cols 0-1
        # Block 3 (C11): rows 2-3, cols 2-3
        r_local = local_idx // 2
        c_local = local_idx % 2
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        r_global = r_start + r_local
        c_global = c_start + c_local
        # W index: k*n + i. Here c_global is k (col), r_global is i (row).
        return c_global * 4 + r_global

    for t in range(7):
        P_blocks = outer_terms[t][0]
        Q_blocks = outer_terms[t][1]
        C_contribs = outer_terms[t][2]
        
        for s in range(7):
            col_idx = t * 7 + s
            
            # Process U (A side)
            for local_a_idx in range(4):
                coeff_inner = U2[local_a_idx, s]
                if coeff_inner != 0:
                    # local_a_idx corresponds to P_t[local_a_idx]
                    # P_t is formed by sum of A blocks
                    # We need to add coeff_inner * (coeff of A in P_t) to U
                    # But P_t[local_a_idx] is the local element.
                    # The mapping is:
                    # P_t is a 2x2 matrix.
                    # P_t[0,0] = sum(A_block[0,0])
                    # P_t[0,1] = sum(A_block[0,1])
                    # etc.
                    # So for a specific local_a_idx (which is r*2+c),
                    # we look at the blocks in P_blocks.
                    # For each block in P_blocks, the element at local position (r,c)
                    # in that block contributes.
                    
                    r_local = local_a_idx // 2
                    c_local = local_a_idx % 2
                    
                    for block_idx, coeff_block in P_blocks:
                        global_a_idx = block_to_global(block_idx, local_a_idx)
                        U[global_a_idx, col_idx] += coeff_inner * coeff_block
            
            # Process V (B side)
            for local_b_idx in range(4):
                coeff_inner = V2[local_b_idx, s]
                if coeff_inner != 0:
                    r_local = local_b_idx // 2
                    c_local = local_b_idx % 2
                    
                    for block_idx, coeff_block in Q_blocks:
                        global_b_idx = block_to_global(block_idx, local_b_idx)
                        V[global_b_idx, col_idx] += coeff_inner * coeff_block
            
            # Process W (C side)
            for local_c_idx in range(4):
                coeff_inner = W2[local_c_idx, s]
                if coeff_inner != 0:
                    for block_idx, coeff_block in C_contribs:
                        global_c_idx = block_to_global_C(block_idx, local_c_idx)
                        W[global_c_idx, col_idx] += coeff_inner * coeff_block
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
