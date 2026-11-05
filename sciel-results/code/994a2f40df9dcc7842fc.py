"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    # Decomposition: M_{3,5,7} = M_{3,2,7} + M_{3,3,7}
    # Rank = 3*2*7 + 3*3*7 = 42 + 63 = 105
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First part: 3x2x7 (columns 0-1 of A, rows 0-1 of B)
    for i in range(3):
        for j in range(2):
            for k in range(7):
                U[i * 5 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    # Second part: 3x3x7 (columns 2-4 of A, rows 2-4 of B)
    for i in range(3):
        for j in range(3):
            for k in range(7):
                U[i * 5 + (j + 2)][col] = 1
                V[(j + 2) * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
