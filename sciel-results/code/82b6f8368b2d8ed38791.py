"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Decompose 4x4 * 4x9 into (4x4 * 4x4) + (4x4 * 4x5)
    # Rank(4x4x4) = 16 (schoolbook)
    # Rank(4x4x5) = 20 (schoolbook)
    # Total Rank = 36
    
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    
    # Part 1: k in 0..3 (4 columns)
    # This handles the 4x4x4 part
    for k in range(4):
        for i in range(n):
            for j in range(m):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Part 2: k in 4..8 (5 columns)
    # This handles the 4x4x5 part
    for k in range(4, 9):
        for i in range(n):
            for j in range(m):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
