import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    # A (2x5) = [AL (2x2), AR (2x3)]
    # B (5x6) = [BL (2x6), BR (3x6)]
    # C = AL BL + AR BR
    
    # Sub-problem 1: AL (2x2) * BL (2x6)
    # BL = [BL1 (2x2), BL2 (2x2), BL3 (2x2)]
    # Each is 2x2x2 (Strassen rank 7)
    # Total rank 21
    
    # Sub-problem 2: AR (2x3) * BR (3x6)
    # BR = [BR1 (3x2), BR2 (3x2), BR3 (3x2)]
    # Each is 2x3x2 (Rank 11)
    # Total rank 33
    
    total_rank = 21 + 33
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # Local indices:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Strassen terms:
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # M2: (A10+A11)B00 -> C10, C11
        # M3: A00 B10 -> C01, C10
        # M4: A11 B11 -> C00, C10
        # M5: (A00+A01)(B11-B10) -> C00, C01
        # M6: (A10-A00)(B00+B01) -> C11
        # M7: (A01-A11)(B10+B11) -> C00, C11
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: [A00, A01, A10, A11]
            # V: [B00, B01, B10, B11]
            # W: [C00, C01, C10, C11] (Note: W index in global is k*n+i, so C00->0, C01->1, C10->2, C11->3 if n=2? 
            # Global W index: k*n + i. 
            # C is n x p. C[i,k]. 
            # Local C is 2x2. 
            # Global mapping: C_local[i, k] -> Global C[c_row_start+i, c_col_start+k].
            # Global W index for C[i,k] is k*n + i.
            # So we need to map local (i,k) to global W index.
            
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            # M3
            ([1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0]),
            # M4
            ([0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, -1, 1], [1, 1, 0, 0]),
            # M6
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            # Add to U
            for li, uj in enumerate(u_coeffs):
                if uj != 0:
                    gi = a_row_start + li // 2
                    gj = a_col_start + li % 2
                    U[gi * m + gj, col_offset] += uj
            # Add to V
            for lj, vk in enumerate(v_coeffs):
                if vk != 0:
                    gj = b_row_start + lj // 2
                    gk = b_col_start + lj % 2
                    V[gj * p + gk, col_offset] += vk
            # Add to W
            for lk, wi in enumerate(w_coeffs):
                if wi != 0:
                    # Local C index (li, lk) where li is row, lk is col
                    # Wait, w_coeffs index:
                    # In Strassen, W corresponds to C.
                    # M1 contributes to C00 and C11.
                    # My w_coeffs for M1 was [1, 0, 0, 1].
                    # Index 0: C00. Index 1: C01. Index 2: C10. Index 3: C11.
                    # So li = index // 2, lk = index % 2.
                    gi = c_row_start + lk // 2 # No, li is row.
                    # Let's re-verify w_coeffs indexing.
                    # C is 2x2. Flattened [C00, C01, C10, C11].
                    # So index 0 -> i=0, k=0.
                    # index 1 -> i=0, k=1.
                    # index 2 -> i=1, k=0.
                    # index 3 -> i=1, k=1.
                    # So li = index // 2? No.
                    # If flattened row-major: C[i, k] -> i * 2 + k.
                    # So i = idx // 2, k = idx % 2.
                    li = lk // 2
                    gk = c_col_start + lk % 2
                    gi = c_row_start + li
                    # Global W index: gk * n + gi
                    W[gk * n + gi, col_offset] += wi
                    
            col_offset += 1
            
    # Helper to add 2x3x2 (Rank 11)
    # A (2x3), B (3x2)
    # Split A into A1 (1x2) and A2 (2x2) ? No, A is 2x3.
    # Split A rows? A is 2 rows.
    # Split A cols? A is 3 cols.
    # Split B rows? B is 3 rows.
    # We used: Split B rows into 1 and 2.
    # B = [B1 (1x2), B2 (2x2)]
    # C = A B1 + A B2
    # A B1: A (2x3) * B1 (1x2) ? No, dimensions mismatch.
    # A is 2x3. B is 3x2.
    # B1 is top 1 row of B: 1x2.
    # A B1 is not defined.
    # We need A (2x3) * B (3x2).
    # Let's split A cols: A = [A1 (2x1), A2 (2x2)].
    # B = [B1 (1x2), B2 (2x2)].
    # C = A1 B1 + A2 B2.
    # A1 B1: (2x1) * (1x2) = 2x2. Rank 4.
    # A2 B2: (2x2) * (2x2) = 2x2. Rank 7.
    # Total 11.
    
    def add_2x3x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # A: 2x3. B: 3x2. C: 2x2.
        # A1: cols 0. A2: cols 1,2.
        # B1: row 0. B2: rows 1,2.
        
        # Part 1: A1 (2x1) * B1 (1x2)
        # Terms:
        # M1: A00 B00 -> C00
        # M2: A00 B01 -> C01
        # M3: A10 B00 -> C10
        # M4: A10 B01 -> C11
        
        terms_1 = [
            ([1, 0, 0], [1, 0], [1, 0, 0, 0]), # A00 B00 -> C00
            ([1, 0, 0], [0, 1], [0, 1, 0, 0]), # A00 B01 -> C01
            ([0, 1, 0], [1, 0], [0, 0, 1, 0]), # A10 B00 -> C10
            ([0, 1, 0], [0, 1], [0, 0, 0, 1]), # A10 B01 -> C11
        ]
        
        # Part 2: A2 (2x2) * B2 (2x2) -> Strassen
        # A2 cols 1,2. B2 rows 1,2.
        
        for u_coeffs, v_coeffs, w_coeffs in terms_1:
            for li, uj in enumerate(u_coeffs):
                if uj != 0:
                    gi = a_row_start + li // 3 # A is 2x3. Flatten: i*3+j.
                    gj = a_col_start + li % 3
                    U[gi * m + gj, col_offset] += uj
            for lj, vk in enumerate(v_coeffs):
                if vk != 0:
                    gj = b_row_start + lj // 2 # B is 3x2. Flatten: j*2+k.
                    gk = b_col_start + lj % 2
                    V[gj * p + gk, col_offset] += vk
            for lk, wi in enumerate(w_coeffs):
                if wi != 0:
                    li = lk // 2
                    gk = c_col_start + lk % 2
                    gi = c_row_start + li
                    W[gk * n + gi, col_offset] += wi
            col_offset += 1
            
        # Add Strassen for A2 B2
        # Map local 2x2 indices to global
        # A2 local (0,0) -> Global (a_row_start, a_col_start+1)
        # A2 local (0,1) -> Global (a_row_start, a_col_start+2)
        # A2 local (1,0) -> Global (a_row_start+1, a_col_start+1)
        # A2 local (1,1) -> Global (a_row_start+1, a_col_start+2)
        
        # We can't directly call add_strassen_2x2x2 because it assumes contiguous 2x2 blocks in global memory?
        # No, it uses offsets.
        # But the local A2 is not a contiguous 2x2 block in the flattened A?
        # A is 2x3. A2 is cols 1,2.
        # Global indices:
        # A2(0,0) -> A(0, a_col_start+1)
        # A2(0,1) -> A(0, a_col_start+2)
        # A2(1,0) -> A(1, a_col_start+1)
        # A2(1,1) -> A(1, a_col_start+2)
        
        # We need a custom Strassen adder for non-contiguous or offset blocks.
        # Or we can just manually define the 7 terms here.
        
        strassen_terms = [
            # U: [A00, A01, A10, A11] (Local A2)
            # V: [B00, B01, B10, B11] (Local B2)
            # W: [C00, C01, C10, C11]
            
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            # M3
            ([1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0]),
            # M4
            ([0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, -1, 1], [1, 1, 0, 0]),
            # M6
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for li, uj in enumerate(u_coeffs):
                if uj != 0:
                    # Local A2 index li (0..3) -> Global
                    # li=0: (0,0) -> (a_row_start, a_col_start+1)
                    # li=1: (0,1) -> (a_row_start, a_col_start+2)
                    # li=2: (1,0) -> (a_row_start+1, a_col_start+1)
                    # li=3: (1,1) -> (a_row_start+1, a_col_start+2)
                    li_row = li // 2
                    li_col = li % 2
                    gi = a_row_start + li_row
                    gj = a_col_start + 1 + li_col
                    U[gi * m + gj, col_offset] += uj
            for lj, vk in enumerate(v_coeffs):
                if vk != 0:
                    # Local B2 index lj (0..3) -> Global
                    # lj=0: (0,0) -> (b_row_start+1, b_col_start)
                    # lj=1: (0,1) -> (b_row_start+1, b_col_start+1)
                    # lj=2: (1,0) -> (b_row_start+2, b_col_start)
                    # lj=3: (1,1) -> (b_row_start+2, b_col_start+1)
                    lj_row = lj // 2
                    lj_col = lj % 2
                    gj = b_row_start + 1 + lj_row
                    gk = b_col_start + lj_col
                    V[gj * p + gk, col_offset] += vk
            for lk, wi in enumerate(w_coeffs):
                if wi != 0:
                    li = lk // 2
                    gk = c_col_start + lk % 2
                    gi = c_row_start + li
                    W[gk * n + gi, col_offset] += wi
            col_offset += 1

    # Call for AL BL1 (2x2x2)
    # A: rows 0-1, cols 0-1
    # B: rows 0-1, cols 0-1
    # C: rows 0-1, cols 0-1
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # Call for AL BL2 (2x2x2)
    # A: rows 0-1, cols 0-1
    # B: rows 0-1, cols 2-3
    # C: rows 0-1, cols 2-3
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    
    # Call for AL BL3 (2x2x2)
    # A: rows 0-1, cols 0-1
    # B: rows 0-1, cols 4-5
    # C: rows 0-1, cols 4-5
    add_strassen_2x2x2(0, 0, 0, 4, 0, 4)
    
    # Call for AR BR1 (2x3x2)
    # A: rows 0-1, cols 2-4
    # B: rows 2-4, cols 0-1
    # C: rows 0-1, cols 0-1
    add_2x3x2(0, 2, 2, 0, 0, 0)
    
    # Call for AR BR2 (2x3x2)
    # A: rows 0-1, cols 2-4
    # B: rows 2-4, cols 2-3
    # C: rows 0-1, cols 2-3
    add_2x3x2(0, 2, 2, 2, 0, 2)
    
    # Call for AR BR3 (2x3x2)
    # A: rows 0-1, cols 2-4
    # B: rows 2-4, cols 4-5
    # C: rows 0-1, cols 4-5
    add_2x3x2(0, 2, 2, 4, 0, 4)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
