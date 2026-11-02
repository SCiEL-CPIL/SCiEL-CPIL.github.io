DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # Format: (U_indices, V_indices, W_indices)
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    terms = [
        # (U, V, W)
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]),   # M1
        ([ (2,1), (3,1) ], [ (0,1) ],       [ (2,1), (3,-1) ]),    # M2
        ([ (0,1) ],       [ (1,1), (3,-1) ], [ (1,1), (3,1) ]),    # M3
        ([ (3,1) ],       [ (2,1), (0,-1) ], [ (0,1), (2,1) ]),    # M4
        ([ (0,1), (1,1) ], [ (3,1) ],       [ (0,-1), (1,1) ]),    # M5
        ([ (2,1), (0,-1) ], [ (0,1), (1,1) ], [ (3,1) ]),          # M6
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (0,1) ]),          # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        # Offsets for V (m*p = 28)
        # V index = j*p + k. j in {0,1}, k in {2b, 2b+1}
        # Local V indices 0,1,2,3 map to:
        # 0: (0, 2b) -> 0*14 + 2b
        # 1: (0, 2b+1) -> 0*14 + 2b + 1
        # 2: (1, 2b) -> 1*14 + 2b
        # 3: (1, 2b+1) -> 1*14 + 2b + 1
        v_off = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # Offsets for W (p*n = 28)
        # W index = k*n + i. k in {2b, 2b+1}, i in {0,1}
        # Local W indices 0,1,2,3 map to:
        # 0: (2b, 0) -> 2b*2 + 0
        # 1: (2b, 1) -> 2b*2 + 1
        # 2: (2b+1, 0) -> (2b+1)*2 + 0
        # 3: (2b+1, 1) -> (2b+1)*2 + 1
        # Wait, the previous code used:
        # w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        # Let's check the convention again.
        # "W[k*n+i, r]"
        # k is column index of C (0..13). i is row index of C (0..1).
        # So index = k*2 + i.
        # For block b, columns are 2b and 2b+1.
        # C[2b, 0] -> 2b*2 + 0 = 4b
        # C[2b, 1] -> 2b*2 + 1 = 4b + 1
        # C[2b+1, 0] -> (2b+1)*2 + 0 = 4b + 2
        # C[2b+1, 1] -> (2b+1)*2 + 1 = 4b + 3
        
        # My local W indices in the terms above were:
        # 0: C00 -> (0,0) of block -> global (2b, 0) -> 4b
        # 1: C01 -> (0,1) of block -> global (2b, 1) -> 4b+1
        # 2: C10 -> (1,0) of block -> global (2b+1, 0) -> 4b+2
        # 3: C11 -> (1,1) of block -> global (2b+1, 1) -> 4b+3
        
        # The previous code had:
        # w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        # This maps local 0->4b, 1->4b+2, 2->4b+1, 3->4b+3.
        # This implies local 1 is C10 and local 2 is C01?
        # Let's check the terms in the previous code.
        # M1: w1 = {0: 1, 3: 1}. C00 and C11.
        # M2: w2 = {2: 1, 3: -1}. C10 and C11.
        # If local 2 is C10, then w_offsets[2] should be index for C10.
        # C10 is (2b+1, 0) -> 4b+2.
        # Previous code w_offsets[2] = 4*b+1.
        # C10 is row 1, col 2b? No.
        # C is 2x14. Rows 0,1. Cols 0..13.
        # C[i, k].
        # W index = k*n + i = k*2 + i.
        # C00 of block b is C[0, 2b]. Index 2b*2 + 0 = 4b.
        # C01 of block b is C[0, 2b+1]. Index (2b+1)*2 + 0 = 4b+2.
        # C10 of block b is C[1, 2b]. Index 2b*2 + 1 = 4b+1.
        # C11 of block b is C[1, 2b+1]. Index (2b+1)*2 + 1 = 4b+3.
        
        # So:
        # Local 0 (C00) -> 4b
        # Local 1 (C01) -> 4b+2
        # Local 2 (C10) -> 4b+1
        # Local 3 (C11) -> 4b+3
        
        # The previous code's w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3] matches this mapping.
        # My new code's w_off should be [4*b, 4*b+2, 4*b+1, 4*b+3].
        
        w_off = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_list, v_list, w_list in terms:
            for idx, val in u_list:
                U[idx][col] += val
            for l_idx, val in v_list:
                V[v_off[l_idx]][col] += val
            for l_idx, val in w_list:
                W[w_off[l_idx]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
