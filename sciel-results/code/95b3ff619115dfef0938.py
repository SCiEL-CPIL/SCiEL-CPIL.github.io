# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # (U_dict, V_dict, W_dict)
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]

    # Define block offsets
    # Block 0: A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    
    # Global index mapping
    # A: i*4 + j
    # B: j*4 + k
    # C: k*2 + i
    
    def map_A(block, li, lj):
        i = li
        j = lj + (2 if block in [1, 3] else 0)
        return i * 4 + j

    def map_B(block, li, lj):
        j = li + (2 if block in [1, 3] else 0)
        k = lj + (2 if block in [2, 3] else 0)
        return j * 4 + k

    def map_C(block, li, lj):
        i = li
        k = lj + (2 if block in [2, 3] else 0)
        return k * 2 + i

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_c, v_c, w_c = terms[s]
            
            for idx, coeff in u_c.items():
                li, lj = idx // 2, idx % 2
                U[map_A(b, li, lj)][col] += coeff
                
            for idx, coeff in v_c.items():
                li, lj = idx // 2, idx % 2
                V[map_B(b, li, lj)][col] += coeff
                
            for idx, coeff in w_c.items():
                li, lj = idx // 2, idx % 2
                W[map_C(b, li, lj)][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
