# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen factors for 2x2x2 block
    # Format: (u_coeff, v_coeff, w_coeff) where coeff is a dict {index: value}
    # Or simpler: lists of (row, coeff)
    
    # Base Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Indices for A (2x2): 00->0, 01->1, 10->2, 11->3
    # Indices for B (2x2): 00->0, 01->1, 10->2, 11->3
    # Indices for C (2x2): 00->0, 01->1, 10->2, 11->3
    
    factors = [
        # (U: {a_idx: val}, V: {b_idx: val}, W: {c_idx: val})
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_dict, v_dict, w_dict = factors[s]
            
            # Fill U
            # A index mapping: i*m + j
            # 0->0, 1->1, 2->2, 3->3 is NOT correct for row-major 2x2?
            # A is 2x2. Row i, Col j. Index i*2+j.
            # 00->0, 01->1, 10->2, 11->3.
            # My dict keys above:
            # 0: A00, 1: A01, 2: A10, 3: A11.
            # This matches i*2+j.
            
            for a_idx, val in u_dict.items():
                U[a_idx][col] = val
            
            # Fill V
            # B index mapping: j*p + k.
            # B is 2x16. But we are in block t.
            # Local B is 2x2. Global B rows 0,1. Cols 2t, 2t+1.
            # Local B indices: 00, 01, 10, 11.
            # Global V row index: j*16 + (2*t + k_local).
            # Local b_idx mapping:
            # 0: B00 (j=0, k=0)
            # 1: B01 (j=0, k=1)
            # 2: B10 (j=1, k=0)
            # 3: B11 (j=1, k=1)
            
            for b_idx, val in v_dict.items():
                j = b_idx // 2
                k_local = b_idx % 2
                global_row = j * p + (2 * t + k_local)
                V[global_row][col] = val
            
            # Fill W
            # C index mapping: k*n + i.
            # C is 16x2.
            # Local C is 2x2.
            # Global C rows: 2t, 2t+1. Cols: 0, 1.
            # Local c_idx mapping:
            # 0: C00 (k=0, i=0)
            # 1: C01 (k=0, i=1)
            # 2: C10 (k=1, i=0)
            # 3: C11 (k=1, i=1)
            
            for c_idx, val in w_dict.items():
                k_local = c_idx // 2
                i = c_idx % 2
                global_row = (2 * t + k_local) * n + i
                W[global_row][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
