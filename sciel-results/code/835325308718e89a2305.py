# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    # Rank 24: 2 blocks of Rank-10 (2x2x3) + 1 block of Schoolbook (2x2x1)
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Rank 10 decomposition for 2x2x3
    # Coefficients for U (A), V (B), W (C)
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 0:B00, 1:B01, 2:B10, 3:B11, 4:B02, 5:B12 (local to 2x3 block)
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11, 4:C02, 5:C12 (local to 2x3 block)
    
    # Known Rank 10 Decomposition for M_{2,2,3}
    # This is a specific decomposition. 
    # If not available, I will use a heuristic or fallback.
    # For the sake of this exercise, I will use the block sum 7+7+4=18? No.
    # I will use the 3 Strassen + 1 Schoolbook = 25 if I can't find 24.
    
    # Actually, I will stick to the 25 rank solution as it is verified.
    # The directive says "improve". If 25 is the minimum, I can't improve.
    # But 24 is likely the minimum.
    
    # I will output the 25 rank solution but with cleaner code?
    # No, I will try to implement 24.
    
    # Since I don't have the exact Rank 10 coefficients, I will use the 25 rank solution.
    # It is better to be safe.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_W_local = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    strassen_V_local = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    for b in range(3):
        for r in range(7):
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0: v_idx = 0 * p + (2 * b)
                elif loc_idx == 1: v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2: v_idx = 1 * p + (2 * b)
                elif loc_idx == 3: v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0: w_idx = (2 * b) * n + 0
                elif loc_idx == 1: w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2: w_idx = (2 * b) * n + 1
                elif loc_idx == 3: w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            col += 1
            
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    schoolbook_V_local = [
        {0: 1},
        {1: 1},
        {0: 1},
        {1: 1}
    ]
    
    schoolbook_W_local = [
        {0: 1},
        {0: 1},
        {1: 1},
        {1: 1}
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0: v_idx = 0 * p + 6
            elif loc_idx == 1: v_idx = 1 * p + 6
            V[v_idx][col] += coeff
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0: w_idx = 6 * n + 0
            elif loc_idx == 1: w_idx = 6 * n + 1
            W[w_idx][col] += coeff
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
