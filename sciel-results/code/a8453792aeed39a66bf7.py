DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U)
    # s: 0, 1, 2, 3, 4, 5, 6
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    U_coeffs = [
        [1, 0, 0, 1], # P0: A00+A11
        [0, 0, 1, 1], # P1: A10+A11
        [1, 0, 0, 0], # P2: A00
        [0, 0, 0, 1], # P3: A11
        [1, 0, 1, 0], # P4: A00+A10
        [-1, 1, 0, 0],# P5: A01-A00
        [0, 0, 1, -1] # P6: A10-A11
    ]
    
    # V and W depend on block r
    for r in range(6):
        base_v0 = 2 * r       # B00
        base_v1 = 2 * r + 1   # B01
        base_v2 = 12 + 2 * r       # B10
        base_v3 = 12 + 2 * r + 1   # B11
        
        base_w0 = 4 * r       # C00
        base_w1 = 4 * r + 1   # C10
        base_w2 = 4 * r + 2   # C01
        base_w3 = 4 * r + 3   # C11
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_coeffs[s][i]
                
            # Fill V
            # V is zero-initialized, so we just set the non-zeros
            if s == 0:
                V[base_v0][c] = 1
                V[base_v3][c] = 1
            elif s == 1:
                V[base_v0][c] = 1
            elif s == 2:
                V[base_v1][c] = 1
                V[base_v3][c] = -1
            elif s == 3:
                V[base_v2][c] = 1
                V[base_v0][c] = -1
            elif s == 4:
                V[base_v3][c] = 1
            elif s == 5:
                V[base_v0][c] = 1
                V[base_v2][c] = 1
            elif s == 6:
                V[base_v1][c] = 1
                V[base_v3][c] = 1
            
            # Fill W
            if s == 0:
                W[base_w0][c] = 1
                W[base_w3][c] = 1
            elif s == 1:
                W[base_w1][c] = 1
                W[base_w3][c] = -1
            elif s == 2:
                W[base_w2][c] = 1
                W[base_w3][c] = 1
            elif s == 3:
                W[base_w0][c] = 1
                W[base_w1][c] = 1
            elif s == 4:
                W[base_w0][c] = -1
                W[base_w2][c] = 1
            elif s == 5:
                W[base_w3][c] = 1
            elif s == 6:
                W[base_w0][c] = -1
                
    return {"U": U, "V": V, "W": W}
