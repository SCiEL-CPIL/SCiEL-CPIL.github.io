"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-term decomposition for 2x2x2
    # M1 = a11 * (b00 - b10)
    # M2 = (a00 + a11) * b10
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b00 + b01)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b11)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # U rows: 0:a00, 1:a01, 2:a10, 3:a11
    # V rows for block t (cols 2t, 2t+1):
    #   2t: b00, 2t+1: b01
    #   16+2t: b10, 16+2t+1: b11
    # W rows for block t (cols 2t, 2t+1):
    #   4t: c00, 4t+1: c10
    #   4t+2: c01, 4t+3: c11
    
    # Coefficients for U (rows 0,1,2,3) for M1..M7
    U_coeffs = [
        [0, 0, 0, 0, 0, 0, 0], # M1
        [0, 0, 0, 0, 0, 0, 0], # M2
        [0, 0, 0, 0, 0, 0, 0], # M3
        [0, 0, 0, 0, 0, 0, 0]  # M4
    ]
    # Wait, let's just define per term
    # Term 0 (M1): a11 -> U[3]=1
    # Term 1 (M2): a00+a11 -> U[0]=1, U[3]=1
    # Term 2 (M3): a00 -> U[0]=1
    # Term 3 (M4): a11 -> U[3]=1
    # Term 4 (M5): a00+a01 -> U[0]=1, U[1]=1
    # Term 5 (M6): a10-a00 -> U[2]=1, U[0]=-1
    # Term 6 (M7): a01-a11 -> U[1]=1, U[3]=-1
    
    U_terms = [
        {3: 1},
        {0: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    # V terms (relative to block start 2t)
    # Term 0 (M1): b00-b10 -> V[0]=1, V[16]=-1
    # Term 1 (M2): b10 -> V[16]=1
    # Term 2 (M3): b01-b11 -> V[1]=1, V[17]=-1
    # Term 3 (M4): b00+b01 -> V[0]=1, V[1]=1
    # Term 4 (M5): b11 -> V[17]=1
    # Term 5 (M6): b00+b11 -> V[0]=1, V[17]=1
    # Term 6 (M7): b10+b11 -> V[16]=1, V[17]=1
    
    V_terms = [
        {0: 1, 16: -1},
        {16: 1},
        {1: 1, 17: -1},
        {0: 1, 1: 1},
        {17: 1},
        {0: 1, 17: 1},
        {16: 1, 17: 1}
    ]
    
    # W terms (relative to block start 4t)
    # c00 = M1 + M4 - M5 + M7 -> W[0] gets 1 for M1, M4, M7 and -1 for M5
    # c10 = M2 + M4 -> W[1] gets 1 for M2, M4
    # c01 = M3 + M5 -> W[2] gets 1 for M3, M5
    # c11 = M1 - M2 + M3 + M6 -> W[3] gets 1 for M1, M3, M6 and -1 for M2
    
    W_terms = [
        {0: 1, 3: 1}, # M1
        {1: 1, 3: -1}, # M2
        {2: 1, 3: 1}, # M3
        {0: 1, 1: 1}, # M4
        {0: -1, 2: 1}, # M5
        {3: 1}, # M6
        {0: 1} # M7
    ]
    
    for t in range(8):
        for s in range(7):
            col = t * 7 + s
            
            for r, val in U_terms[s].items():
                U[r][col] = val
                
            base_v = 2 * t
            for r, val in V_terms[s].items():
                V[base_v + r][col] = val
                
            base_w = 4 * t
            for r, val in W_terms[s].items():
                W[base_w + r][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
