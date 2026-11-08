"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper to add a rank-1 term
    # u_indices: list of (i, j, coeff)
    # v_indices: list of (j, k, coeff)
    # w_indices: list of (k, i, coeff)
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # Split A (4x4) into 2x2 blocks: A00, A01, A10, A11
    # Split B (4x4) into 2x2 blocks: B00, B01, B10, B11
    # Indices:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # Same for B (first 4 cols)
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # Strassen 7 multiplications for 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product.
    # We decompose each 2x2 product using Strassen (7 scalar mults).
    
    # Define the 7 Strassen combinations for the inner 2x2
    # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    
    # Result C = [[s1+s4-s5+s7, s3+s5], [s2+s4, s1-s2+s3+s6]]
    
    # We need to map the "super" Strassen terms to the inner Strassen terms.
    
    # Super Term 1: P = A00+A11, Q = B00+B11
    # P is 2x2. P[0,0] = A[0,0]+A[2,2], P[0,1] = A[0,1]+A[2,3], etc.
    # Q is 2x2. Q[0,0] = B[0,0]+B[2,2], etc.
    
    # Function to get U, V, W lists for a super term
    def get_super_term_indices(super_idx, inner_idx):
        # Returns u_list, v_list, w_list for the specific scalar multiplication
        # super_idx: 0..6
        # inner_idx: 0..6
        
        # Define the linear combinations for P and Q for each super term
        # P is sum of blocks of A. Q is sum of blocks of B.
        
        # Block definitions:
        # A00: i<2, j<2. A01: i<2, j>=2. A10: i>=2, j<2. A11: i>=2, j>=2.
        # B00: j<2, k<2. B01: j<2, k>=2. B10: j>=2, k<2. B11: j>=2, k>=2.
        
        # Coefficients for A blocks in P
        # Coefficients for B blocks in Q
        
        # We'll build u_list and v_list by expanding the linear combinations.
        
        u_list = []
        v_list = []
        w_list = []
        
        # Determine P and Q structure based on super_idx
        # P blocks: (coeff_A00, coeff_A01, coeff_A10, coeff_A11)
        # Q blocks: (coeff_B00, coeff_B01, coeff_B10, coeff_B11)
        
        if super_idx == 0: # M1
            p_blocks = [1, 0, 0, 1] # A00 + A11
            q_blocks = [1, 0, 0, 1] # B00 + B11
        elif super_idx == 1: # M2
            p_blocks = [0, 0, 1, 1] # A10 + A11
            q_blocks = [1, 0, 0, 0] # B00
        elif super_idx == 2: # M3
            p_blocks = [1, 0, 0, 0] # A00
            q_blocks = [0, 1, 0, -1] # B01 - B11
        elif super_idx == 3: # M4
            p_blocks = [0, 0, 0, 1] # A11
            q_blocks = [-1, 0, 1, 0] # B10 - B00
        elif super_idx == 4: # M5
            p_blocks = [1, 1, 0, 0] # A00 + A01
            q_blocks = [0, 0, 0, 1] # B11
        elif super_idx == 5: # M6
            p_blocks = [-1, 0, 1, 0] # A10 - A00
            q_blocks = [1, 1, 0, 0] # B00 + B01
        elif super_idx == 6: # M7
            p_blocks = [0, 1, 0, -1] # A01 - A11
            q_blocks = [0, 0, 1, 1] # B10 + B11
        else:
            return u_list, v_list, w_list
            
        # Now, for the inner Strassen term inner_idx, determine which elements of P and Q are multiplied
        # P = [[P00, P01], [P10, P11]]
        # Q = [[Q00, Q01], [Q10, Q11]]
        
        # Inner Strassen terms:
        # 0: (P00+P11)(Q00+Q11)
        # 1: (P10+P11)Q00
        # 2: P00(Q01-Q11)
        # 3: P11(Q10-Q00)
        # 4: (P00+P01)Q11
        # 5: (P10-P00)(Q00+Q01)
        # 6: (P01-P11)(Q10+Q11)
        
        # We need to identify the linear combination of P elements and Q elements.
        
        # Helper to get P element linear combo
        # P_ab is a 2x2 matrix.
        # P_ab[0,0] corresponds to A block element at local (0,0)
        # Local (0,0) of A00 is global (0,0).
        # Local (0,1) of A00 is global (0,1).
        # Local (1,0) of A00 is global (1,0).
        # Local (1,1) of A00 is global (1,1).
        # Local (0,0) of A01 is global (0,2).
        # ...
        
        def get_p_element(p_elem_idx):
            # p_elem_idx: 0:00, 1:01, 2:10, 3:11
            # Returns list of (i, j, coeff)
            res = []
            # Determine which block and local position
            # p_elem_idx 0: P00. P00 is top-left 2x2 of P.
            # P is formed by adding blocks.
            # P00[0,0] = sum(coeff_block * A_block[0,0])
            # P00[0,1] = sum(coeff_block * A_block[0,1])
            
            # We need to sum over all 4 blocks.
            # Block 0 (A00): local pos (0,0) -> global (0,0)
            # Block 1 (A01): local pos (0,0) -> global (0,2)
            # Block 2 (A10): local pos (0,0) -> global (2,0)
            # Block 3 (A11): local pos (0,0) -> global (2,2)
            
            # p_elem_idx maps to local row r, col c in the 2x2 P matrix.
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                # Global i, j for block b_idx at local (r,c)
                if b_idx == 0: # A00
                    gi = r; gj = c
                elif b_idx == 1: # A01
                    gi = r; gj = 2 + c
                elif b_idx == 2: # A10
                    gi = 2 + r; gj = c
                elif b_idx == 3: # A11
                    gi = 2 + r; gj = 2 + c
                
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            # q_elem_idx: 0:00, 1:01, 2:10, 3:11
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: # B00
                    gj = r; gk = c
                elif b_idx == 1: # B01
                    gj = r; gk = 2 + c
                elif b_idx == 2: # B10
                    gj = 2 + r; gk = c
                elif b_idx == 3: # B11
                    gj = 2 + r; gk = 2 + c
                
                res.append((gj, gk, coeff))
            return res
            
        # Determine P and Q elements for this inner term
        # inner_idx 0: (P00+P11)(Q00+Q11)
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)] # P00, P11
            q_elems = [(0, 1), (3, 1)] # Q00, Q11
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)] # P10, P11
            q_elems = [(0, 1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)] # Q01, -Q11
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)] # Q10, -Q00
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)] # P00, P01
            q_elems = [(3, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)] # P10, -P00
            q_elems = [(0, 1), (1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)] # P01, -P11
            q_elems = [(2, 1), (3, 1)]
        else:
            return u_list, v_list, w_list
            
        # Build u_list
        for p_idx, p_coeff in p_elems:
            elems = get_p_element(p_idx)
            for i, j, c in elems:
                u_list.append((i, j, c * p_coeff))
                
        # Build v_list
        for q_idx, q_coeff in q_elems:
            elems = get_q_element(q_idx)
            for j, k, c in elems:
                v_list.append((j, k, c * q_coeff))
                
        # Build w_list
        # The result of this scalar multiplication contributes to C.
        # We need to determine the coefficients for C[i,k].
        # The inner Strassen result S is a 2x2 matrix.
        # S = P Q.
        # The contribution of this inner term to S is known.
        # Then S contributes to the final C (4x4) based on the super term position.
        
        # This part is complex. Let's use the known Strassen coefficients.
        # For inner term t, the contribution to S[0,0], S[0,1], S[1,0], S[1,1] is:
        
        if inner_idx == 0:
            s_contrib = [1, 0, 0, 1] # S00, S01, S10, S11
        elif inner_idx == 1:
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2:
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3:
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4:
            s_contrib = [-1, 1, 0, 1]
        elif inner_idx == 5:
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6:
            s_contrib = [1, 0, 0, 1] # Wait, check Strassen formula
        else:
            s_contrib = [0,0,0,0]
            
        # Correct Strassen C formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for inner term 0 (M1): contrib to C00:1, C11:1
        # Inner 1 (M2): C10:1, C11:-1
        # Inner 2 (M3): C01:1, C11:1
        # Inner 3 (M4): C00:1, C10:1
        # Inner 4 (M5): C00:-1, C01:1
        # Inner 5 (M6): C11:1
        # Inner 6 (M7): C00:1
        
        # Map inner_idx to S contributions
        if inner_idx == 0: s_contrib = [1, 0, 0, 1]
        elif inner_idx == 1: s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2: s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3: s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4: s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5: s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6: s_contrib = [1, 0, 0, 0]
        
        # Now map S contributions to C (4x4) contributions.
        # S is a 2x2 block of C.
        # Super term 0 (M1) corresponds to C00 block?
        # No, the 7 super terms are combined to form the 4 C blocks.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for a given super_idx, we know which C block it contributes to and with what sign.
        # super_idx 0 (M1): C00 (+1), C11 (+1)
        # super_idx 1 (M2): C10 (+1), C11 (-1)
        # super_idx 2 (M3): C01 (+1), C11 (+1)
        # super_idx 3 (M4): C00 (+1), C10 (+1)
        # super_idx 4 (M5): C00 (-1), C01 (+1)
        # super_idx 5 (M6): C11 (+1)
        # super_idx 6 (M7): C00 (+1)
        
        c_block_contrib = [0, 0, 0, 0] # C00, C01, C10, C11
        if super_idx == 0:
            c_block_contrib = [1, 0, 0, 1]
        elif super_idx == 1:
            c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2:
            c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3:
            c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4:
            c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5:
            c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6:
            c_block_contrib = [1, 0, 0, 0]
            
        # Combine s_contrib and c_block_contrib
        # s_contrib is for S[0,0], S[0,1], S[1,0], S[1,1]
        # c_block_contrib is for C00, C01, C10, C11
        # S is the result of P Q.
        # The elements of S are added to C blocks.
        # S[0,0] adds to C00. S[0,1] adds to C00? No.
        # S is a 2x2 matrix.
        # C00 is a 2x2 block.
        # So S[0,0] contributes to C[0,0]. S[0,1] to C[0,1]. S[1,0] to C[1,0]. S[1,1] to C[1,1].
        # This is for the C00 block.
        # If the super term contributes to C01, then S[0,0] contributes to C[0,2], etc.
        
        # Let's define the mapping from (c_block, s_pos) to global (i, k)
        # c_block 0: C00. i in 0,1; k in 0,1
        # c_block 1: C01. i in 0,1; k in 2,3
        # c_block 2: C10. i in 2,3; k in 0,1
        # c_block 3: C11. i in 2,3; k in 2,3
        
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff == 0: continue
            # s_pos 0: S[0,0]. 1: S[0,1]. 2: S[1,0]. 3: S[1,1]
            s_r = s_pos // 2
            s_c = s_pos % 2
            
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff == 0: continue
                
                total_coeff = s_coeff * c_coeff
                if total_coeff == 0: continue
                
                # Determine global i, k
                if c_b == 0:
                    gi = s_r; gk = s_c
                elif c_b == 1:
                    gi = s_r; gk = 2 + s_c
                elif c_b == 2:
                    gi = 2 + s_r; gk = s_c
                elif c_b == 3:
                    gi = 2 + s_r; gk = 2 + s_c
                    
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    # Add all 49 terms
    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_super_term_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1

    # --- Part 2: 4x4 * 4x1 using Schoolbook ---
    # A (4x4) * B_col (4x1)
    # B_col is column 4 of B.
    # C_col is column 4 of C.
    # C[i,4] = sum_j A[i,j] B[j,4]
    
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
