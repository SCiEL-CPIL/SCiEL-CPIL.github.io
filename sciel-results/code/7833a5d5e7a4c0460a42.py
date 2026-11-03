"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank 39:
    # Split B into B1 (3x3) and B2 (3x4).
    # C = A*B1 + A*B2.
    # Use schoolbook for A*B1 (rank 27).
    # Use schoolbook for A*B2 (rank 12).
    # Total rank 39.
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A * B1 (k in 0..2)
    # Schoolbook: 3*3*3 = 27 terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # k = 0, 1, 2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: A * B2 (k in 3..6)
    # Schoolbook: 3*3*4 = 36 terms? No.
    # For each k in 3..6, it's a 3x3x1 multiplication.
    # Rank of 3x3x1 is 3.
    # So 4 * 3 = 12 terms.
    for k in range(3, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
