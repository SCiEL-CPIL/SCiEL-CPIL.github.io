"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9

    # Rank decomposition of the 2x3x9 matrix multiplication tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Split the middle dimension m=3 into m1=2 and m2=1
    # Part 1: 2x2x9 tensor (rank 9) - use Kronecker product structure
    # Part 2: 2x1x9 tensor (rank 1) - use all-ones structure
    # Total rank: 9 + 1 = 10
    
    rank = 10
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 10)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 10)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 10)
    
    # Part 1: Handle j in [0, 2) (first 2 rows of A)
    # For each k in [0, 9), we need to cover i in [0, 2) and j in [0, 2)
    # Use rank 9 terms, one for each k
    for k in range(p):
        c = k  # column index
        # U[i*3 + j][c] = 1 for i in [0,2), j in [0,2)
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = 1
        # V[j*9 + k][c] = 1 for j in [0,2)
        for j in range(2):
            V[j * p + k][c] = 1
        # W[k*2 + i][c] = 1 for i in [0,2)
        for i in range(2):
            W[k * n + i][c] = 1
    
    # Part 2: Handle j = 2 (third row of A)
    # For all i in [0,2), k in [0,9), we need one term
    c = 9  # column index
    # U[i*3 + 2][c] = 1 for i in [0,2)
    for i in range(2):
        U[i * m + 2][c] = 1
    # V[2*9 + k][c] = 1 for k in [0,9)
    for k in range(9):
        V[2 * p + k][c] = 1
    # W[k*2 + i][c] = 1 for k in [0,9), i in [0,2)
    for k in range(9):
        for i in range(2):
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
