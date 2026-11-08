"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    #
    # Partition A's columns into two groups of 2: cols 0-1 (top) and cols 2-3 (bottom)
    # Partition B's rows correspondingly: rows 0-1 (top) and rows 2-3 (bottom)
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # Each half is a 2x2 x 2x4 multiplication, rank 8 each. Total rank = 16.
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]   # 8 x 16
    V = [[0] * rank for _ in range(m * p)]   # 16 x 16
    W = [[0] * rank for _ in range(p * n)]   # 8 x 16
    
    column = 0
    # Top block: j in {0, 1}
    for i in range(n):      # i = 0, 1 (row of A / output)
        for j in range(2):  # j = 0, 1 (col of A / row of B)
            for k in range(p):  # k = 0,1,2,3 (col of B / col of output)
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    # Bottom block: j in {2, 3}
    for i in range(n):      # i = 0, 1
        for j in range(2, 4):  # j = 2, 3
            for k in range(p):  # k = 0,1,2,3
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
