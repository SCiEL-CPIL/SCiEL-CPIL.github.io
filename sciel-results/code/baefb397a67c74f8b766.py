"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # Split B into B1 (4x4) and B2 (4x1)
    # C = [A B1, A B2]
    # Rank for A B1 is 4*4 = 16
    # Rank for A B2 is 4*1 = 4? No, A is 4x4, B2 is 4x1.
    # C2 is 4x1.
    # C2[i, 0] = sum_j A[i, j] B2[j, 0]
    # Terms: i in 1..4, j in 1..4. Total 16 terms.
    # So total rank is 16 + 16 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A B1 (first 4 columns of C)
    for i in range(n):
        for j in range(m):
            for k in range(4): # k is 0..3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: A B2 (last column of C)
    for i in range(n):
        for j in range(m):
            k = 4 # k is 4 (0-indexed)
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
