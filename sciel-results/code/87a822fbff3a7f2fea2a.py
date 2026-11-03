"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank is 7 * (3*3*3) = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for X_t (A blocks)
    # t=0..6 corresponding to Strassen terms 1..7
    # Blocks: 00, 01, 10, 11
    X_coeff = [
        [1, 0, 0, 1], # t=0: A00 + A11
        [0, 0, 1, 1], # t=1: A10 + A11
        [1, 0, 0, 0], # t=2: A00
        [0, 0, 0, 1], # t=3: A11
        [1, 1, 0, 0], # t=4: A00 + A01
        [-1, 0, 1, 0],# t=5: A10 - A00
        [0, 1, 0, -1] # t=6: A01 - A11
    ]
    
    # Coefficients for Y_t (B blocks)
    Y_coeff = [
        [1, 0, 0, 1], # t=0: B00 + B11
        [1, 0, 0, 0], # t=1: B00
        [0, 1, 0, -1],# t=2: B01 - B11
        [-1, 0, 1, 0],# t=3: B10 - B00
        [0, 0, 0, 1], # t=4: B11
        [1, 1, 0, 0], # t=5: B00 + B01
        [0, 0, 1, 1]  # t=6: B10 + B11
    ]
    
    # Coefficients for W_t (C blocks receiving Z_t)
    # C00 = Z0 + Z3 - Z4 + Z6
    # C01 = Z2 + Z4
    # C10 = Z1 + Z3
    # C11 = Z0 - Z1 + Z2 + Z5
    W_coeff = [
        [1, 0, 0, 1], # t=0: C00(1), C11(1)
        [0, 0, 1, -1],# t=1: C10(1), C11(-1)
        [0, 1, 0, 1], # t=2: C01(1), C11(1)
        [1, 0, 1, 0], # t=3: C00(1), C10(1)
        [-1, 1, 0, 0],# t=4: C00(-1), C01(1)
        [0, 0, 0, 1], # t=5: C11(1)
        [1, 0, 0, 0]  # t=6: C00(1)
    ]
    
    col = 0
    for t in range(7):
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # Fill U
                    for rc in range(4):
                        coeff = X_coeff[t][rc]
                        if coeff != 0:
                            r = rc // 2
                            c = rc % 2
                            i = 3 * r + u
                            j = 3 * c + v
                            U[i * 6 + j][col] = coeff
                    
                    # Fill V
                    for rc in range(4):
                        coeff = Y_coeff[t][rc]
                        if coeff != 0:
                            r = rc // 2
                            c = rc % 2
                            j = 3 * r + v
                            k = 3 * c + w
                            V[j * 6 + k][col] = coeff
                    
                    # Fill W
                    for rc in range(4):
                        coeff = W_coeff[t][rc]
                        if coeff != 0:
                            r = rc // 2
                            c = rc % 2
                            # C block rc has local row u, local col w
                            # Global i = 3r + u, Global k = 3c + w
                            # W index is k * n + i
                            k = 3 * c + w
                            i = 3 * r + u
                            W[k * 6 + i][col] = coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
