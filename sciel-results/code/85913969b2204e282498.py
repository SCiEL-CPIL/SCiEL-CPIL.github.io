"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose B into three parts: 4 cols, 4 cols, 1 col.
    # Each part is multiplied by A (4x4).
    # Part 1: 4x4 * 4x4 -> Rank 16
    # Part 2: 4x4 * 4x4 -> Rank 16
    # Part 3: 4x4 * 4x1 -> Rank 16
    # Total Rank = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Group 1: Columns 0-3 of B
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Group 2: Columns 4-7 of B
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Group 3: Column 8 of B
    for i in range(n):
        for j in range(m):
            for k in range(8, 9):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
