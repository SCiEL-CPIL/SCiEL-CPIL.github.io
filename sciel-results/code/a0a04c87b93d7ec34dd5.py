"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 (Strassen) * 9 (3x3 schoolbook) = 63
    rank = 63
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrices
    # Indices: row is (a,b) or (b,c) or (c,a) flattened 0..3
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    # U_strassen: coeff of A_ab in M_s
    # s=0: A00+A11 -> [1, 0, 0, 1]
    # s=1: A10+A11 -> [0, 0, 1, 1]
    # s=2: A00     -> [1, 0, 0, 0]
    # s=3: A11     -> [0, 0, 0, 1]
    # s=4: A00+A01 -> [1, 1, 0, 0]
    # s=5: A10-A00 -> [-1, 0, 1, 0]
    # s=6: A01-A11 -> [0, 1, 0, -1]
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V_strassen: coeff of B_bc in M_s
    # s=0: B00+B11 -> [1, 0, 0, 1]
    # s=1: B00     -> [1, 0, 0, 0]
    # s=2: B10-B11 -> [0, 0, 1, -1]
    # s=3: B01-B10 -> [0, 1, -1, 0]
    # s=4: B11     -> [0, 0, 0, 1]
    # s=5: B00+B01 -> [1, 1, 0, 0]
    # s=6: B10+B11 -> [0, 0, 1, 1]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W_strassen: coeff of C_ca in sum of M_s
    # s=0: C00+C11 -> [1, 0, 0, 1]
    # s=1: C10-C11 -> [0, 0, 1, -1]
    # s=2: C01+C11 -> [0, 1, 0, 1]
    # s=3: C00+C10 -> [1, 0, 1, 0]
    # s=4: -C00+C01 -> [-1, 1, 0, 0]
    # s=5: C11      -> [0, 0, 0, 1]
    # s=6: C00      -> [1, 0, 0, 0]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Iterate over 3x3 block structure
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Iterate over 7 Strassen terms
                for s in range(7):
                    col = ((I * 3 + J) * 3 + K) * 7 + s
                    
                    # Fill U
                    # i = 2I + a, j = 2J + b
                    for a in range(2):
                        for b in range(2):
                            i = 2 * I + a
                            j = 2 * J + b
                            row_u = i * m + j
                            # Index into U_strassen: row is (a,b), col is s
                            # (a,b) index: a*2 + b
                            val_u = U_strassen[s][a * 2 + b]
                            if val_u != 0:
                                U[row_u][col] = val_u
                                
                    # Fill V
                    # j = 2J + b, k = 2K + c
                    for b in range(2):
                        for c in range(2):
                            j = 2 * J + b
                            k = 2 * K + c
                            row_v = j * p + k
                            # Index into V_strassen: row is (b,c), col is s
                            # (b,c) index: b*2 + c
                            val_v = V_strassen[s][b * 2 + c]
                            if val_v != 0:
                                V[row_v][col] = val_v
                                
                    # Fill W
                    # k = 2K + c, i = 2I + a
                    for c in range(2):
                        for a in range(2):
                            k = 2 * K + c
                            i = 2 * I + a
                            row_w = k * n + i
                            # Index into W_strassen: row is (c,a), col is s
                            # (c,a) index: c*2 + a
                            val_w = W_strassen[s][c * 2 + a]
                            if val_w != 0:
                                W[row_w][col] = val_w
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
