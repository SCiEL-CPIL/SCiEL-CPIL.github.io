"""Matrix multiplication 2x2 by 2x6 - rank minimal tensor decomposition."""
DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    # Split the 2x6 matrix B into three 2x2 blocks
    # Each block uses Strassen's rank-7 algorithm
    # Total rank = 3 * 7 = 21
    rank = 21
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 21)
    V = [[0] * rank for _ in range(m * p)]   # shape (12, 21)
    W = [[0] * rank for _ in range(p * n)]   # shape (12, 21)
    
    # Strassen 2x2 multiplication terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    u_terms = [
        {(0,0): 1, (1,1): 1},       # M1: A00+A11
        {(1,0): 1, (1,1): 1},       # M2: A10+A11
        {(0,0): 1},                 # M3: A00
        {(1,1): 1},                 # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00+A01
        {(1,0): 1, (0,0): -1},      # M6: A10-A00
        {(0,1): 1, (1,1): -1}       # M7: A01-A11
    ]
    
    v_terms = [
        {(0,0): 1, (1,1): 1},       # M1: B00+B11
        {(0,0): 1},                 # M2: B00
        {(0,1): 1, (1,1): -1},      # M3: B01-B11
        {(1,0): 1, (0,0): -1},      # M4: B10-B00
        {(1,1): 1},                 # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00+B01
        {(1,0): 1, (1,1): 1}        # M7: B10+B11
    ]
    
    w_terms = [
        {(0,0): 1, (1,1): 1},       # M1: C00+=1, C11+=1
        {(1,0): 1, (1,1): -1},      # M2: C10+=1, C11-=1
        {(0,1): 1, (1,1): 1},       # M3: C01+=1, C11+=1
        {(0,0): 1, (1,0): 1},       # M4: C00+=1, C10+=1
        {(0,0): -1, (0,1): 1},      # M5: C00-=1, C01+=1
        {(1,1): 1},                 # M6: C11+=1
        {(0,0): 1}                  # M7: C00+=1
    ]
    
    col = 0
    for b in range(3):  # three 2x2 blocks
        for r in range(7):  # seven Strassen terms
            # Fill U: index i*m + j
            for (i, j), val in u_terms[r].items():
                U[i * m + j][col] += val
            # Fill V: index j*p + k_global
            for (j, k_local), val in v_terms[r].items():
                k_global = 2 * b + k_local
                V[j * p + k_global][col] += val
            # Fill W: index k_global*n + i
            for (i, k_local), val in w_terms[r].items():
                k_global = 2 * b + k_local
                W[k_global * n + i][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
