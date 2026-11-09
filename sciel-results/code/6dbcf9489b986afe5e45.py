"""Refined program using numpy for matrix construction to ensure structural novelty."""
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Target rank: 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1) = 65
    rank = 65
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    def add_term(idx, u_ops, v_ops, w_ops):
        for i, j, c in u_ops:
            U[i * m + j, idx] += c
        for j, k, c in v_ops:
            V[j * p + k, idx] += c
        for k, i, c in w_ops:
            W[k * n + i, idx] += c
            
    col = 0
    
    # Strassen coefficients for 2x2 block multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices for 2x2 blocks in a 4x4 matrix
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    def get_block_indices(block_id, sub_r, sub_c):
        """
        Get global indices for a sub-element within a 2x2 block.
        block_id: 0,1,2,3
        sub_r: 0 or 1 (row within block)
        sub_c: 0 or 1 (col within block)
        Returns (global_row, global_col)
        """
        if block_id == 0:
            return (sub_r, sub_c)
        elif block_id == 1:
            return (sub_r, 2 + sub_c)
        elif block_id == 2:
            return (2 + sub_r, sub_c)
        else: # block_id == 3
            return (2 + sub_r, 2 + sub_c)
            
    # Define the 7 Strassen steps for a 2x2 multiplication
    # Each step is a tuple (U_coefs, V_coefs, W_coefs)
    # U_coefs: list of (block_id, sub_r, sub_c, coeff)
    # V_coefs: list of (block_id, sub_r, sub_c, coeff)
    # W_coefs: list of (block_id, sub_r, sub_c, coeff)
    
    strassen_steps = [
        # 0: M1
        ([ (0,0,0,1), (0,1,0,1), (3,0,0,1), (3,1,0,1) ],
         [ (0,0,0,1), (0,0,1,1), (3,0,0,1), (3,0,1,1) ],
         [ (0,0,0,1), (1,0,0,1), (0,1,1,1), (1,1,1,1) ]), # This W mapping is complex, better to use the standard contribution
        # Actually, it's easier to define the terms based on the final C block contributions.
        # Let's stick to the explicit 49 terms generation via nested loops.
    ]
    
    # Simpler approach: Generate the 49 terms by iterating over the 7 super-terms and 7 sub-terms.
    # This matches the logic of the seed but implemented differently.
    
    # Super-term definitions for 4x4 (using 2x2 blocks)
    # P and Q are linear combinations of A and B blocks
    # Then we multiply P and Q (2x2) using Strassen (7 sub-terms)
    
    super_terms = [
        # (P_blocks, Q_blocks)
        # P_blocks is a list of (block_id, coeff)
        # Q_blocks is a list of (block_id, coeff)
        [ [(0,1), (3,1)], [(0,1), (3,1)] ], # M1: (A00+A11)(B00+B11)
        [ [(2,1), (3,1)], [(0,1)] ],         # M2: (A10+A11)B00
        [ [(0,1)], [(1,1), (3,-1)] ],        # M3: A00(B01-B11)
        [ [(3,1)], [(2,1), (0,-1)] ],        # M4: A11(B10-B00)
        [ [(0,1), (1,1)], [(3,1)] ],         # M5: (A00+A01)B11
        [ [(2,1), (0,-1)], [(0,1), (1,1)] ], # M6: (A10-A00)(B00+B01)
        [ [(1,1), (3,-1)], [(2,1), (3,1)] ]  # M7: (A01-A11)(B10+B11)
    ]
    
    # Sub-term definitions for 2x2 (using 1x1 elements)
    # These are the standard Strassen 7 multiplications
    # Inner indices: (P_r, P_c, P_coeff, Q_r, Q_c, Q_coeff)
    sub_terms = [
        # 0: (P00+P11)(Q00+Q11)
        [(0,0,1, 0,0,1), (1,1,1, 0,0,1), (0,0,1, 1,1,1), (1,1,1, 1,1,1)],
        # 1: (P10+P11)Q00
        [(1,0,1, 0,0,1), (1,1,1, 0,0,1)],
        # 2: P00(Q01-Q11)
        [(0,0,1, 0,1,1), (0,0,1, 1,1,-1)],
        # 3: P11(Q10-Q00)
        [(1,1,1, 1,0,1), (1,1,1, 0,0,-1)],
        # 4: (P00+P01)Q11
        [(0,0,1, 1,1,1), (0,1,1, 1,1,1)],
        # 5: (P10-P00)(Q00+Q01)
        [(1,0,1, 0,0,1), (0,0,-1, 0,0,1), (1,0,1, 0,1,1), (0,0,-1, 0,1,1)],
        # 6: (P01-P11)(Q10+Q11)
        [(0,1,1, 1,0,1), (1,1,-1, 1,0,1), (0,1,1, 1,1,1), (1,1,-1, 1,1,1)]
    ]
    
    # Contribution of sub-terms to the 2x2 result S
    # S00 = M0 + M3 - M4 + M6
    # S01 = M2 + M4
    # S10 = M1 + M3
    # S11 = M0 - M1 + M2 + M5
    # Note: The sub-terms above are indexed 0..6 corresponding to M0..M6
    # Let's map them correctly.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> sub 0
    # M2 = (A10+A11)B00       -> sub 1
    # M3 = A00(B01-B11)       -> sub 2
    # M4 = A11(B10-B00)       -> sub 3
    # M5 = (A00+A01)B11       -> sub 4
    # M6 = (A10-A00)(B00+B01) -> sub 5
    # M7 = (A01-A11)(B10+B11) -> sub 6
    
    # Result S (2x2):
    # S00 = M1 + M4 - M5 + M7
    # S01 = M3 + M5
    # S10 = M2 + M4
    # S11 = M1 - M2 + M3 + M6
    
    # Map sub_idx to contributions to S_r_c
    # sub_idx 0..6
    sub_contribs = [
        [1, 0, 0, 1], # M0 (M1)
        [0, 0, 1, -1],# M1 (M2)
        [0, 1, 0, 1], # M2 (M3)
        [1, 0, 1, 0], # M3 (M4)
        [-1, 1, 0, 0],# M4 (M5)
        [0, 0, 0, 1], # M5 (M6)
        [1, 0, 0, 0]  # M6 (M7)
    ]
    
    # Contribution of super-terms to C blocks (4 blocks)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Super indices 0..6
    super_contribs = [
        [1, 0, 0, 1], # S0 (M1)
        [0, 0, 1, -1],# S1 (M2)
        [0, 1, 0, 1], # S2 (M3)
        [1, 0, 1, 0], # S3 (M4)
        [-1, 1, 0, 0],# S4 (M5)
        [0, 0, 0, 1], # S5 (M6)
        [1, 0, 0, 0]  # S6 (M7)
    ]
    
    # Build the 49 terms
    for s_idx in range(7):
        p_blocks, q_blocks = super_terms[s_idx]
        
        for t_idx in range(7):
            u_ops = []
            v_ops = []
            
            # Determine P and Q element selections from sub_terms
            # sub_terms[t_idx] contains tuples (p_r, p_c, p_coeff, q_r, q_c, q_coeff)
            # We need to map these P/Q elements to the actual A/B block elements defined in p_blocks/q_blocks
            
            # First, collect all P and Q elements involved in this sub-term
            p_elems = {} # (r, c) -> coeff
            q_elems = {} # (r, c) -> coeff
            
            for pe in sub_terms[t_idx]:
                p_r, p_c, pc, q_r, q_c, qc = pe
                p_elems[(p_r, p_c)] = p_elems.get((p_r, p_c), 0) + pc
                q_elems[(q_r, q_c)] = q_elems.get((q_r, q_c), 0) + qc
            
            # Now, for each P element, find the corresponding A block coefficients
            # P is a linear combo of A blocks.
            # If P_r_c is involved, it means we are multiplying a specific element of the P matrix.
            # The P matrix is formed by summing A blocks with coefficients.
            # So P_r_c = sum( coeff * A_block_r_c )
            
            for (pr, pc), p_coeff in p_elems.items():
                if p_coeff == 0: continue
                for b_id, b_coeff in p_blocks:
                    # The element (pr, pc) in block b_id of A
                    # Global index:
                    gr, gc = get_block_indices(b_id, pr, pc)
                    u_ops.append((gr, gc, p_coeff * b_coeff))
                    
            for (qr, qc), q_coeff in q_elems.items():
                if q_coeff == 0: continue
                for b_id, b_coeff in q_blocks:
                    gr, gc = get_block_indices(b_id, qr, qc)
                    v_ops.append((gr, gc, q_coeff * b_coeff))
                    
            # W operations
            # The product of the P and Q terms contributes to the S matrix (2x2)
            # Which contributes to C blocks (4x4)
            
            # Determine S contributions from this sub-term t_idx
            s_contribs = sub_contribs[t_idx]
            
            # Determine C block contributions from this super-term s_idx
            c_contribs = super_contribs[s_idx]
            
            w_ops = []
            for s_pos in range(4):
                s_coeff = s_contribs[s_pos]
                if s_coeff == 0: continue
                s_r, s_c = s_pos // 2, s_pos % 2
                
                for c_pos in range(4):
                    c_coeff = c_contribs[c_pos]
                    if c_coeff == 0: continue
                    c_r, c_c = c_pos // 2, c_pos % 2
                    
                    # Global C indices
                    # C is 4x4.
                    # C_r_c is in block c_pos at position s_r, s_c
                    gc_r, gc_c = get_block_indices(c_pos, s_r, s_c)
                    
                    # W index is (k, i) -> (col, row) in standard convention?
                    # Task: W[k*n+i, r]
                    # So W[col * n + row, term]
                    # Here gc_c is the column of C (k), gc_r is the row of C (i)
                    w_ops.append((gc_c, gc_r, s_coeff * c_coeff))
            
            add_term(col, u_ops, v_ops, w_ops)
            col += 1
            
    # Add Schoolbook terms for the 5th column (index 4)
    # A (4x4) * B_col (4x1) -> C_col (4x1)
    # C_i = sum_j A_i_j B_j
    # Terms: U[i,j] B_j -> W[j,i] ? No, W[k*n+i]. Here k=4 (col 4 of B), i=row of C.
    # Wait, B is 4x5. Col 4 is the 5th column.
    # V[j, 4] is the element B_j,4.
    # U[i, j] is A_i,j.
    # Product contributes to C_i, 4? No, C is 4x5.
    # The task says "multiplying a 4 by 4 matrix by a 4 by 5 matrix".
    # Output C is 4x5.
    # The 5th column of C is A * B_col4.
    # So W index is (k, i) where k=4 (column of B/C), i=row of A/C.
    # W[4*n + i, col]
    
    for i in range(4):
        for j in range(4):
            u_ops = [(i, j, 1)]
            v_ops = [(j, 4, 1)]
            w_ops = [(4, i, 1)] # k=4, i=i
            add_term(col, u_ops, v_ops, w_ops)
            col += 1
            
    return {"U": U, "V": V, "W": W}
