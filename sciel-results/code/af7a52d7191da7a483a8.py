"""Rewritten program using Strassen for 2x2 blocks and schoolbook for remainders."""
import numpy as np

def construct():
    n, m, p = 3, 3, 6
    
    # Dimensions
    # A: 3x3 (9)
    # B: 3x6 (18)
    # C: 3x6 (18)
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices):
        """
        Adds a term to the decomposition.
        u_indices: dict {a_index: coeff}
        v_indices: dict {b_index: coeff}
        w_indices: dict {c_index: coeff}
        """
        nonlocal rank
        rank += 1
        u_row = [0] * 9
        v_row = [0] * 18
        w_row = [0] * 18
        
        for idx, val in u_indices.items():
            u_row[idx] = val
        for idx, val in v_indices.items():
            v_row[idx] = val
        for idx, val in w_indices.items():
            w_row[idx] = val
            
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
        
    # Indices
    # A: i*m + j
    # 0 1 2
    # 3 4 5
    # 6 7 8
    #
    # B: j*p + k
    # 0 1 2 3 4 5
    # 6 7 8 9 10 11
    # 12 13 14 15 16 17
    #
    # C: k*n + i
    # 0 1 2 3 4 5
    # 6 7 8 9 10 11
    # 12 13 14 15 16 17
    
    # M1: A11 * B11 (2x2 * 2x6)
    # A11 indices: 0,1,3,4
    # B11 indices: 0..11 (rows 0,1)
    # C_top indices: 0..11 (rows 0,1)
    
    # Strassen for 2x2x2
    # M1 = (A11+A21)(B11+B21)
    # M2 = A21(B11+B21)
    # M3 = A11 B11
    # M4 = A22 B21
    # M5 = (A11+A22)(B21-B11)
    # M6 = (A11-A21)(B11+B12)
    # M7 = (A21+A22)(B21-B11)
    
    # Note: Standard Strassen indices for A = [[A11, A12], [A21, A22]]
    # Here A11 is the 2x2 block.
    # Let's use local indices 0,1,2,3 for A11.
    # A11[0,0]=A0, A11[0,1]=A1, A11[1,0]=A3, A11[1,1]=A4.
    
    # We process B11 in 3 chunks of 2 columns.
    # Chunk 0: Cols 0,1. B indices 0,1 (row 0), 6,7 (row 1).
    # Chunk 1: Cols 2,3. B indices 2,3, 8,9.
    # Chunk 2: Cols 4,5. B indices 4,5, 10,11.
    
    for s in range(3):
        # B chunk indices
        b_r0_c0 = s*2      # e.g. 0
        b_r0_c1 = s*2 + 1  # e.g. 1
        b_r1_c0 = 6 + s*2  # e.g. 6
        b_r1_c1 = 6 + s*2 + 1 # e.g. 7
        
        # C chunk indices (Top 2 rows)
        c_r0_c0 = s*2
        c_r0_c1 = s*2 + 1
        c_r1_c0 = 6 + s*2
        c_r1_c1 = 6 + s*2 + 1
        
        # Strassen Terms
        # 1
        add_term({0: 1, 3: 1}, {b_r0_c0: 1, b_r1_c0: 1, b_r0_c1: 1, b_r1_c1: 1}, 
                 {c_r0_c0: 1, c_r0_c1: -1, c_r1_c0: -1, c_r1_c1: 1}) # M1 - M5 + M7 - M4? 
        # Wait, the W matrix coefficients must match the C formula.
        # C11 = M1 - M5 + M7 - M4
        # C12 = M3 + M5
        # C21 = M6 + M4
        # C22 = M1 - M2 + M3 - M6
        
        # I will add 7 terms per chunk.
        
        # Term 1: M1 = (A0+A3)(B_r0_c0+B_r1_c0+B_r0_c1+B_r1_c1)
        # Actually, Strassen inputs:
        # M1 = (A11+A21) * (B11+B21)
        # A11=A0, A21=A3.
        # B11=B_r0_c0, B21=B_r1_c0.
        # Note: In Strassen, B is 2x2.
        # B11 is top-left, B21 is bottom-left.
        # For chunk s, B is 2x2.
        # B[0,0]=b_r0_c0, B[0,1]=b_r0_c1, B[1,0]=b_r1_c0, B[1,1]=b_r1_c1.
        
        # M1 = (A0+A3) * (b_r0_c0 + b_r1_c0)
        add_term({0: 1, 3: 1}, {b_r0_c0: 1, b_r1_c0: 1}, 
                 {c_r0_c0: 1, c_r1_c1: 1}) # M1 contributes to C11 and C22
        
        # M2 = A3 * (b_r0_c0 + b_r1_c0)
        add_term({3: 1}, {b_r0_c0: 1, b_r1_c0: 1}, 
                 {c_r1_c1: -1}) # M2 contributes to -C22
        
        # M3 = A0 * b_r0_c0
        add_term({0: 1}, {b_r0_c0: 1}, 
                 {c_r0_c1: 1, c_r1_c1: 1}) # M3 contributes to C12 and C22
        
        # M4 = A4 * b_r1_c1
        add_term({4: 1}, {b_r1_c1: 1}, 
                 {c_r0_c0: -1, c_r1_c0: 1}) # M4 contributes to -C11 and C21
        
        # M5 = (A0+A4) * (b_r1_c1 - b_r0_c0)
        add_term({0: 1, 4: 1}, {b_r1_c1: 1, b_r0_c0: -1}, 
                 {c_r0_c0: -1, c_r0_c1: 1}) # M5 contributes to -C11 and C12
        
        # M6 = (A0-A3) * (b_r0_c0 + b_r0_c1)
        add_term({0: 1, 3: -1}, {b_r0_c0: 1, b_r0_c1: 1}, 
                 {c_r1_c0: 1, c_r1_c1: -1}) # M6 contributes to C21 and -C22
        
        # M7 = (A3+A4) * (b_r1_c1 - b_r0_c0)
        add_term({3: 1, 4: 1}, {b_r1_c1: 1, b_r0_c0: -1}, 
                 {c_r0_c0: 1}) # M7 contributes to C11

    # M2: A12 * B21 (2x1 * 1x6)
    # A12 indices: 2, 5
    # B21 indices: 12..17
    # C_top (rows 0,1) cols 0..5
    for k in range(6):
        b_idx = 12 + k
        c_r0 = k
        c_r1 = 6 + k
        add_term({2: 1, 5: 0}, {b_idx: 1}, {c_r0: 1}) # A2 * Bk -> C0k
        add_term({2: 0, 5: 1}, {b_idx: 1}, {c_r1: 1}) # A5 * Bk -> C1k

    # M3: A21 * B11 (1x2 * 2x6)
    # A21 indices: 6, 7
    # B11 indices: 0..11
    # C_bot (row 2) cols 0..5
    for k in range(6):
        b_r0 = k
        b_r1 = 6 + k
        c_r2 = 12 + k
        add_term({6: 1, 7: 0}, {b_r0: 1}, {c_r2: 1}) # A6 * Br0 -> C2k
        add_term({6: 0, 7: 1}, {b_r1: 1}, {c_r2: 1}) # A7 * Br1 -> C2k

    # M4: A22 * B21 (1x1 * 1x6)
    # A22 index: 8
    # B21 indices: 12..17
    # C_bot (row 2) cols 0..5
    for k in range(6):
        b_idx = 12 + k
        c_r2 = 12 + k
        add_term({8: 1}, {b_idx: 1}, {c_r2: 1})

    return {"U": U, "V": V, "W": W}
