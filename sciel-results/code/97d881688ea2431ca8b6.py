import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # U: 16 x rank
    # V: 36 x rank
    # W: 36 x rank
    
    # Strassen 2x2 sub-terms
    # u, v, w are length 4
    # Indices 0,1,2,3 correspond to 00, 01, 10, 11
    strassen_2x2 = [
        # M1: (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (X10+X11)Y00 -> Z10+=1, Z11-=-1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: X00(Y01-Y11) -> Z01+=1, Z11+=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4: X11(Y10-Y00) -> Z00+=1, Z10+=1
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5: (X00+X01)Y11 -> Z00-=-1, Z01+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (X10-X00)(Y00+Y01) -> Z11+=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (X01-X11)(Y10+Y11) -> Z00+=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Strassen 4x4 meta-terms
    # alpha: coeffs for A blocks (00, 01, 10, 11)
    # beta: coeffs for B blocks (00, 01, 10, 11)
    # gamma: coeffs for C blocks (00, 01, 10, 11)
    strassen_4x4_meta = [
        # M1: (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10+=M2, C11-=M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: A00(B01-B11) -> C01+=M3, C11+=M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4: A11(B10-B00) -> C00+=M4, C10+=M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5: (A00+A01)B11 -> C00-=M5, C01+=M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A10-A00)(B00+B01) -> C11+=M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00+=M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 49 + 49 + 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(col, u_vals, v_vals, w_vals, u_offset=0, v_offset=0, w_offset=0, u_scale=1.0, v_scale=1.0, w_scale=1.0):
        # u_vals, v_vals, w_vals are lists of length 4 (for 2x2 blocks)
        # We need to expand them to 16 values for 4x4
        pass

    # Helper to expand 2x2 block coeffs to 4x4 vector
    def expand_2x2_to_4x4(vec_2x2, block_coeffs):
        # vec_2x2: [c00, c01, c10, c11]
        # block_coeffs: [b00, b01, b10, b11]
        # Result: 16 values
        res = [0.0] * 16
        # Block 00: rows 0,1 cols 0,1 -> indices 0,1,4,5
        # Block 01: rows 0,1 cols 2,3 -> indices 2,3,6,7
        # Block 10: rows 2,3 cols 0,1 -> indices 8,9,12,13
        # Block 11: rows 2,3 cols 2,3 -> indices 10,11,14,15
        
        # Mapping from 2x2 index (r,c) to 4x4 indices for each block
        # Block 00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
        # Block 01: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
        # Block 10: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
        # Block 11: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
        
        offsets = [
            [0, 1, 4, 5],   # Block 00
            [2, 3, 6, 7],   # Block 01
            [8, 9, 12, 13], # Block 10
            [10, 11, 14, 15]# Block 11
        ]
        
        for b_idx in range(4):
            b_coeff = block_coeffs[b_idx]
            if b_coeff == 0: continue
            for rc_idx in range(4):
                v = vec_2x2[rc_idx]
                if v == 0: continue
                idx = offsets[b_idx][rc_idx]
                res[idx] += v * b_coeff
        return res

    # Part 1: C1 = A B1 (cols 0-3 of B)
    # B1 corresponds to cols 0-3.
    # In V, col index k is 0-3.
    # In W, col index k is 0-3.
    
    for m_idx, (alpha, beta, gamma) in enumerate(strassen_4x4_meta):
        for s_idx, (u2, v2, w2) in enumerate(strassen_2x2):
            col = m_idx * 7 + s_idx
            
            u_full = expand_2x2_to_4x4(u2, alpha)
            v_full = expand_2x2_to_4x4(v2, beta)
            w_full = expand_2x2_to_4x4(w2, gamma)
            
            for i in range(16):
                if u_full[i] != 0:
                    U[i][col] += u_full[i]
                if v_full[i] != 0:
                    # V index: j * p + k. Here k is 0-3.
                    # v_full[i] corresponds to B1 element i.
                    # B1 is 4x4. i maps to j, k.
                    # i = j * 4 + k.
                    # V index = j * 9 + k.
                    j = i // 4
                    k = i % 4
                    v_idx = j * 9 + k
                    V[v_idx][col] += v_full[i]
                if w_full[i] != 0:
                    # W index: k * n + i. Here k is 0-3.
                    # w_full[i] corresponds to C1 element i.
                    # C1 is 4x4. i maps to row, col.
                    # i = row * 4 + col.
                    # W index = col * 4 + row.
                    row = i // 4
                    col_c = i % 4
                    w_idx = col_c * 4 + row
                    W[w_idx][col] += w_full[i]

    # Part 2: C2 = A B2 (cols 4-7 of B)
    # B2 corresponds to cols 4-7.
    # In V, col index k is 4-7.
    # In W, col index k is 4-7.
    
    for m_idx, (alpha, beta, gamma) in enumerate(strassen_4x4_meta):
        for s_idx, (u2, v2, w2) in enumerate(strassen_2x2):
            col = 49 + m_idx * 7 + s_idx
            
            u_full = expand_2x2_to_4x4(u2, alpha)
            v_full = expand_2x2_to_4x4(v2, beta)
            w_full = expand_2x2_to_4x4(w2, gamma)
            
            for i in range(16):
                if u_full[i] != 0:
                    U[i][col] += u_full[i]
                if v_full[i] != 0:
                    j = i // 4
                    k = i % 4
                    # B2 col is 4+k
                    v_idx = j * 9 + (4 + k)
                    V[v_idx][col] += v_full[i]
                if w_full[i] != 0:
                    row = i // 4
                    col_c = i % 4
                    # C2 col is 4+col_c
                    w_idx = (4 + col_c) * 4 + row
                    W[w_idx][col] += w_full[i]

    # Part 3: C3 = A B3 (col 8 of B)
    # Rank 16
    for i in range(4):
        for j in range(4):
            col = 98 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0

    return {"U": U, "V": V, "W": W}
