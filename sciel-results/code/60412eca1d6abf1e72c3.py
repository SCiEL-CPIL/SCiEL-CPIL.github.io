"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using 3x3 blocks of 2x2 Strassen
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for A (U), B (V), C (W)
    # Local indices 0,1
    # M1: A00+A11, B00+B11 -> C00+C11
    # M2: A10+A11, B00      -> C10-C11
    # M3: A00,      B01-B11 -> C01+C11
    # M4: A11,      B10-B00 -> C00+C10
    # M5: A00+A01,  B11     -> -C00+C01
    # M6: A10-A00,  B00+B01 -> C11
    # M7: A01-A11,  B10+B11 -> C00
    
    # Define the patterns for U, V, W
    # Each pattern is a list of (row_idx, col_idx, coeff)
    # We use 0-based local indices for the 2x2 block
    
    U_patterns = [
        [(0, 0, 1), (1, 1, 1)],       # M1
        [(1, 0, 1), (1, 1, 1)],       # M2
        [(0, 0, 1)],                  # M3
        [(1, 1, 1)],                  # M4
        [(0, 0, 1), (0, 1, 1)],       # M5
        [(1, 0, 1), (0, 0, -1)],      # M6
        [(0, 1, 1), (1, 1, -1)],      # M7
    ]
    
    V_patterns = [
        [(0, 0, 1), (1, 1, 1)],       # M1
        [(0, 0, 1)],                  # M2
        [(0, 1, 1), (1, 1, -1)],      # M3
        [(1, 0, 1), (0, 0, -1)],      # M4
        [(1, 1, 1)],                  # M5
        [(0, 0, 1), (0, 1, 1)],       # M6
        [(1, 0, 1), (1, 1, 1)],       # M7
    ]
    
    W_patterns = [
        [(0, 0, 1), (1, 1, 1)],       # M1
        [(1, 0, 1), (1, 1, -1)],      # M2
        [(0, 1, 1), (1, 1, 1)],       # M3
        [(0, 0, 1), (1, 0, 1)],       # M4
        [(0, 0, -1), (0, 1, 1)],      # M5
        [(1, 1, 1)],                  # M6
        [(0, 0, 1)],                  # M7
    ]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for t in range(7):
                    # U: A[I,J] block
                    for lr, lc, coeff in U_patterns[t]:
                        r = 2 * I + lr
                        c = 2 * J + lc
                        U[r * m + c][col] = coeff
                    
                    # V: B[J,K] block
                    for lr, lc, coeff in V_patterns[t]:
                        r = 2 * J + lr
                        c = 2 * K + lc
                        V[r * p + c][col] = coeff
                    
                    # W: C[I,K] block
                    for lr, lc, coeff in W_patterns[t]:
                        r = 2 * I + lr
                        c = 2 * K + lc
                        W[r * n + c][col] = coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
