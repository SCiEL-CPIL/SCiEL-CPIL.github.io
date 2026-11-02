"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm parameters
    # A indices: 0: a11, 1: a12, 2: a21, 3: a22
    # B indices for block t (cols 2t, 2t+1):
    #   b11: 2t
    #   b12: 2t+1
    #   b21: 12+2t
    #   b22: 12+2t+1
    # C indices for block t:
    #   c11 (row 0, col 2t): 4t
    #   c12 (row 0, col 2t+1): 4t+2
    #   c21 (row 1, col 2t): 4t+1
    #   c22 (row 1, col 2t+1): 4t+3
    
    for t in range(6):
        col_base = 7 * t
        
        # M1 = a11 * (b22 - b12)
        c = col_base
        U[0][c] = 1
        V[12 + 2*t + 1][c] = 1
        V[2*t + 1][c] = -1
        W[4*t + 2][c] = 1
        W[4*t + 3][c] = 1
        
        # M2 = (a11 + a12) * b22
        c = col_base + 1
        U[0][c] = 1
        U[1][c] = 1
        V[12 + 2*t + 1][c] = 1
        W[4*t][c] = -1
        W[4*t + 2][c] = 1
        
        # M3 = (a21 + a22) * b11
        c = col_base + 2
        U[2][c] = 1
        U[3][c] = 1
        V[2*t][c] = 1
        W[4*t + 1][c] = 1
        W[4*t + 3][c] = -1
        
        # M4 = a22 * (b12 - b11)
        c = col_base + 3
        U[3][c] = 1
        V[2*t + 1][c] = 1
        V[2*t][c] = -1
        W[4*t][c] = 1
        W[4*t + 1][c] = 1
        
        # M5 = (a11 + a22) * (b11 + b22)
        c = col_base + 4
        U[0][c] = 1
        U[3][c] = 1
        V[2*t][c] = 1
        V[12 + 2*t + 1][c] = 1
        W[4*t][c] = 1
        W[4*t + 3][c] = 1
        
        # M6 = (a12 - a22) * (b21 + b22)
        c = col_base + 5
        U[1][c] = 1
        U[3][c] = -1
        V[12 + 2*t][c] = 1
        V[12 + 2*t + 1][c] = 1
        W[4*t][c] = 1
        
        # M7 = (a11 - a21) * (b11 + b12)
        c = col_base + 6
        U[0][c] = 1
        U[2][c] = -1
        V[2*t][c] = 1
        V[2*t + 1][c] = 1
        W[4*t + 3][c] = -1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
