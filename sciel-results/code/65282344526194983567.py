"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 3 blocks of 2x2
    for b in range(3):
        base_col = 2 * b
        for s in range(7):
            r = b * 7 + s
            
            if s == 0:
                # M1 = (a11 + a22)(b11 + b22)
                U[0][r] = 1
                U[3][r] = 1
                V[0 * 6 + base_col][r] = 1
                V[1 * 6 + base_col + 1][r] = 1
                W[base_col * 2 + 0][r] = 1
                W[(base_col + 1) * 2 + 1][r] = 1
                
            elif s == 1:
                # M2 = (a21 + a22)b11
                U[2][r] = 1
                U[3][r] = 1
                V[0 * 6 + base_col][r] = 1
                W[base_col * 2 + 1][r] = 1
                W[(base_col + 1) * 2 + 1][r] = -1
                
            elif s == 2:
                # M3 = a11(b12 - b22)
                U[0][r] = 1
                V[0 * 6 + base_col + 1][r] = 1
                V[1 * 6 + base_col + 1][r] = -1
                W[base_col * 2 + 0][r] = -1
                W[(base_col + 1) * 2 + 0][r] = 1
                
            elif s == 3:
                # M4 = a22(b21 - b11)
                U[3][r] = 1
                V[1 * 6 + base_col][r] = 1
                V[0 * 6 + base_col][r] = -1
                W[base_col * 2 + 0][r] = 1
                W[base_col * 2 + 1][r] = 1
                
            elif s == 4:
                # M5 = (a11 + a12)b22
                U[0][r] = 1
                U[1][r] = 1
                V[1 * 6 + base_col + 1][r] = 1
                W[(base_col + 1) * 2 + 0][r] = 1
                W[(base_col + 1) * 2 + 1][r] = 1
                
            elif s == 5:
                # M6 = (a21 - a11)(b11 + b12)
                U[2][r] = 1
                U[0][r] = -1
                V[0 * 6 + base_col][r] = 1
                V[0 * 6 + base_col + 1][r] = 1
                W[base_col * 2 + 1][r] = 1
                W[(base_col + 1) * 2 + 1][r] = 1
                
            elif s == 6:
                # M7 = (a12 - a22)(b21 + b22)
                U[1][r] = 1
                U[3][r] = -1
                V[1 * 6 + base_col][r] = 1
                V[1 * 6 + base_col + 1][r] = 1
                W[base_col * 2 + 0][r] = 1
                W[(base_col + 1) * 2 + 0][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
