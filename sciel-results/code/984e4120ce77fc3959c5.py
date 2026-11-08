"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B = [A*B1 | A*B2 | A*B3]
    # A*B1: 4x4 x 4x4, schoolbook rank 64
    # A*B2: 4x4 x 4x4, schoolbook rank 64
    # A*B3: 4x4 x 4x1, rank 1
    # Total rank: 64 + 64 + 1 = 129
    
    rank = 129
    U = [[0] * rank for _ in range(n * m)]  # 16 x 129
    V = [[0] * rank for _ in range(m * p)]  # 36 x 129
    W = [[0] * rank for _ in range(p * n)]  # 36 x 129
    
    col = 0
    # Part 1: A*B1 (columns 0-3 of C)
    for i in range(n):
        for j in range(m):
            for k in range(4):  # k=0..3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: A*B2 (columns 4-7 of C)
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):  # k=4..7
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 3: A*B3 (column 8 of C), rank 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
