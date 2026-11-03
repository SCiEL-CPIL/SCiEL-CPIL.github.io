# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Indices: 0:00, 1:01, 2:10, 3:11
    # Term s: (U_dict, V_dict, W_dict)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]
    
    # Block mapping
    # Blocks: 0: LL, 1: RL, 2: LR, 3: RR
    # A: 2x4 -> split into 2x2 blocks A_L (cols 0,1), A_R (cols 2,3)
    # B: 4x4 -> split into 2x2 blocks B_LL (rows 0,1 cols 0,1), B_RL (rows 2,3 cols 0,1), B_LR (rows 0,1 cols 2,3), B_RR (rows 2,3 cols 2,3)
    # C: 2x4 -> split into 2x2 blocks C_L (cols 0,1), C_R (cols 2,3)
    
    # Global indices
    # A[i, j] -> i*4 + j
    # B[j, r] -> j*4 + r
    # C[i, r] -> r*2 + i (Trace convention: W[k*n+i, r] -> here k=r, n=2, i=i. So index is r*2+i)
    
    def map_A(block, li, lj):
        # li: row (0,1), lj: col (0,1)
        i = li
        if block in [0, 2]: # A_L
            j = lj
        else: # A_R
            j = 2 + lj
        return i * 4 + j
        
    def map_B(block, li, lj):
        # li: row (0,1), lj: col (0,1)
        # B is 4x4.
        # Block 0 (LL): rows 0-1, cols 0-1
        # Block 1 (RL): rows 2-3, cols 0-1
        # Block 2 (LR): rows 0-1, cols 2-3
        # Block 3 (RR): rows 2-3, cols 2-3
        if block == 0:
            j = li
            r = lj
        elif block == 1:
            j = 2 + li
            r = lj
        elif block == 2:
            j = li
            r = 2 + lj
        elif block == 3:
            j = 2 + li
            r = 2 + lj
        return j * 4 + r
        
    def map_C(block, li, lj):
        # li: row (0,1), lj: col (0,1)
        # C is 2x4.
        # Block 0,1: C_L (cols 0,1)
        # Block 2,3: C_R (cols 2,3)
        i = li
        if block in [0, 1]:
            r = lj
        else:
            r = 2 + lj
        return r * 2 + i

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = map_A(b, li, lj)
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = map_B(b, li, lj)
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = map_C(b, li, lj)
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
