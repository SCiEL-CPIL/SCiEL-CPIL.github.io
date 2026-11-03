import time

def _make_strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2.
    U: 4x7, V: 4x7, W: 4x7.
    U index: i*2 + j
    V index: j*2 + k
    W index: k*2 + i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Terms: (U_coeffs, V_coeffs, W_coeffs)
    # Indices: U: 0:00, 1:01, 2:10, 3:11
    #          V: 0:00, 1:01, 2:10, 3:11
    #          W: 0:00, 1:01, 2:10, 3:11
    
    terms = [
        # M1: (A00+A11)(B00+B11)
        # U: A00(0)=1, A11(3)=1. V: B00(0)=1, B11(3)=1.
        # W: C00(0)+=1, C01(1)+=1, C10(2)-=1, C11(3)+=1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),
        
        # M2: (A10+A11)B00
        # U: A10(2)=1, A11(3)=1. V: B00(0)=1.
        # W: C10(2)+=1, C11(3)-=1
        ({2:1, 3:1}, {0:1}, {0:0, 1:0, 2:1, 3:-1}),
        
        # M3: A00(B01-B11)
        # U: A00(0)=1. V: B01(1)=1, B11(3)=-1.
        # W: C01(1)+=1, C10(2)+=1
        ({0:1}, {1:1, 3:-1}, {0:0, 1:1, 2:1, 3:0}),
        
        # M4: A11(B10-B00)
        # U: A11(3)=1. V: B10(2)=1, B00(0)=-1.
        # W: C00(0)-=1, C10(2)+=1
        ({3:1}, {2:1, 0:-1}, {0:-1, 1:0, 2:1, 3:0}),
        
        # M5: (A00+A10)B11
        # U: A00(0)=1, A10(2)=1. V: B11(3)=1.
        # W: C00(0)+=1, C01(1)-=1
        ({0:1, 2:1}, {3:1}, {0:1, 1:-1, 2:0, 3:0}),
        
        # M6: (A10-A00)(B00+B10)
        # U: A10(2)=1, A00(0)=-1. V: B00(0)=1, B10(2)=1.
        # W: C00(0)+=1, C11(3)+=1
        ({2:1, 0:-1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),
        
        # M7: (A01-A11)(B01+B11)
        # U: A01(1)=1, A11(3)=-1. V: B01(1)=1, B11(3)=1.
        # W: C01(1)+=1, C11(3)+=1
        ({1:1, 3:-1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),
    ]
    
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][t] = val
        for idx, val in v_c.items():
            V[idx][t] = val
        for idx, val in w_c.items():
            W[idx][t] = val
            
    return U, V, W

def _make_strassen_4x4():
    """
    Returns U, V, W for 4x4 * 4x4 -> 4x4.
    Rank 49.
    U: 16x49, V: 16x49, W: 16x49.
    U index: i*4 + j
    V index: j*4 + k
    W index: k*4 + i
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    u2, v2, w2 = _make_strassen_2x2()
    
    # Block offsets for 2x2 blocks within 4x4
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    # Strassen terms for 4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each M is a 2x2 * 2x2 multiplication.
    # We need to map the block indices to the 2x2 Strassen indices.
    
    # For U (A):
    # A00 is block (0,0) -> U indices 0..3 (i in 0..1, j in 0..1)
    # A01 is block (0,1) -> U indices 2..5? No.
    # U index is i*4+j.
    # Block (br, bc) covers i in [br*2, br*2+2), j in [bc*2, bc*2+2).
    # Local index in block: li in 0..1, lj in 0..1.
    # Global index: (br*2+li)*4 + (bc*2+lj).
    # In the 2x2 Strassen U, the index is li*2+lj.
    
    # Let's define the mapping for each block.
    def map_u(block_r, block_c, local_idx):
        li, lj = divmod(local_idx, 2)
        gi = block_r * 2 + li
        gj = block_c * 2 + lj
        return gi * 4 + gj
        
    def map_v(block_r, block_c, local_idx):
        # V index is j*4+k.
        # Block (br, bc) covers j in [br*2, br*2+2), k in [bc*2, bc*2+2).
        # Local index: lj, lk.
        # In 2x2 Strassen V, index is lj*2+lk.
        lj, lk = divmod(local_idx, 2)
        gj = block_r * 2 + lj
        gk = block_c * 2 + lk
        return gj * 4 + gk
        
    def map_w(block_r, block_c, local_idx):
        # W index is k*4+i.
        # Block (br, bc) covers k in [br*2, br*2+2), i in [bc*2, bc*2+2).
        # Local index: lk, li.
        # In 2x2 Strassen W, index is lk*2+li.
        lk, li = divmod(local_idx, 2)
        gk = block_r * 2 + lk
        gi = block_c * 2 + li
        return gk * 4 + gi

    # Strassen combinations
    # M1: A blocks {0,3}, B blocks {0,3}, C blocks {0,1,2,3} with signs
    # M2: A blocks {2,3}, B blocks {0}, C blocks {2,3} with signs
    # M3: A blocks {0}, B blocks {1,3} with signs, C blocks {1,2} with signs
    # M4: A blocks {3}, B blocks {2,0} with signs, C blocks {0,2} with signs
    # M5: A blocks {0,2}, B blocks {3}, C blocks {0,1} with signs
    # M6: A blocks {2,0} with signs, B blocks {0,2} with signs, C blocks {0,3} with signs
    # M7: A blocks {1,3} with signs, B blocks {1,3} with signs, C blocks {1,3} with signs
    
    # We iterate over the 7 Strassen terms for 4x4.
    # For each term, we have a set of A-blocks with coefficients, B-blocks with coefficients, C-blocks with coefficients.
    # Then for each 2x2 term (0..6) within those blocks, we add the corresponding 2x2 Strassen entries.
    
    # Let's define the 4x4 Strassen terms as lists of (block, coeff)
    # A_blocks: list of (block_id, coeff)
    # B_blocks: list of (block_id, coeff)
    # C_blocks: list of (block_id, coeff)
    
    strassen_4x4_terms = [
        # M1
        (
            [(0, 1), (3, 1)],  # A: A00 + A11
            [(0, 1), (3, 1)],  # B: B00 + B11
            [(0, 1), (1, 1), (2, -1), (3, 1)] # C: C00 + C01 - C10 + C11
        ),
        # M2
        (
            [(2, 1), (3, 1)],  # A: A10 + A11
            [(0, 1)],          # B: B00
            [(2, 1), (3, -1)]  # C: C10 - C11
        ),
        # M3
        (
            [(0, 1)],          # A: A00
            [(1, 1), (3, -1)], # B: B01 - B11
            [(1, 1), (2, 1)]   # C: C01 + C10
        ),
        # M4
        (
            [(3, 1)],          # A: A11
            [(2, 1), (0, -1)], # B: B10 - B00
            [(0, -1), (2, 1)]  # C: -C00 + C10
        ),
        # M5
        (
            [(0, 1), (2, 1)],  # A: A00 + A10
            [(3, 1)],          # B: B11
            [(0, 1), (1, -1)]  # C: C00 - C01
        ),
        # M6
        (
            [(2, 1), (0, -1)], # A: A10 - A00
            [(0, 1), (2, 1)],  # B: B00 + B10
            [(0, 1), (3, 1)]   # C: C00 + C11
        ),
        # M7
        (
            [(1, 1), (3, -1)], # A: A01 - A11
            [(1, 1), (3, 1)],  # B: B01 + B11
            [(1, 1), (3, 1)]   # C: C01 + C11
        ),
    ]
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_4x4_terms):
        # For each 2x2 term in the 2x2 Strassen
        for t2 in range(7):
            global_col = t_idx * 7 + t2
            # Get the 2x2 Strassen coefficients for this term
            # u2, v2, w2 are 4x7. Column t2 contains the coeffs.
            u2_col = [u2[i][t2] for i in range(4)]
            v2_col = [v2[i][t2] for i in range(4)]
            w2_col = [w2[i][t2] for i in range(4)]
            
            # Apply to A blocks
            for block_id, coeff in a_blocks:
                br, bc = divmod(block_id, 2)
                for local_idx in range(4):
                    val = u2_col[local_idx] * coeff
                    if val != 0:
                        g_idx = map_u(br, bc, local_idx)
                        U[g_idx][global_col] += val
                        
            # Apply to B blocks
            for block_id, coeff in b_blocks:
                br, bc = divmod(block_id, 2)
                for local_idx in range(4):
                    val = v2_col[local_idx] * coeff
                    if val != 0:
                        g_idx = map_v(br, bc, local_idx)
                        V[g_idx][global_col] += val
                        
            # Apply to C blocks
            for block_id, coeff in c_blocks:
                br, bc = divmod(block_id, 2)
                for local_idx in range(4):
                    val = w2_col[local_idx] * coeff
                    if val != 0:
                        g_idx = map_w(br, bc, local_idx)
                        W[g_idx][global_col] += val
                        
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    
    # Get 4x4 Strassen
    U44, V44, W44 = _make_strassen_4x4()
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First half (cols 0-3 of B and C)
    for i in range(16):
        for r in range(49):
            U[i][r] = U44[i][r]
            
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + k
            v44_idx = j * 4 + k
            for r in range(49):
                V[v_idx][r] = V44[v44_idx][r]
                
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            w44_idx = k * 4 + i
            for r in range(49):
                W[w_idx][r] = W44[w44_idx][r]
                
    # Second half (cols 4-7 of B and C)
    for i in range(16):
        for r in range(49):
            U[i][r + 49] = U44[i][r]
            
    for j in range(4):
        for k in range(4, 8):
            v_idx = j * 8 + k
            k_local = k - 4
            v44_idx = j * 4 + k_local
            for r in range(49):
                V[v_idx][r + 49] = V44[v44_idx][r]
                
    for k in range(4, 8):
        for i in range(4):
            w_idx = k * 4 + i
            k_local = k - 4
            w44_idx = k_local * 4 + i
            for r in range(49):
                W[w_idx][r + 49] = W44[w44_idx][r]
                
    return {"U": U, "V": V, "W": W}
