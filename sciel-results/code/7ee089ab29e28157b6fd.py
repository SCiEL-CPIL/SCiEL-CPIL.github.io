"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-6 decomposition for 2x3x7 matrix multiplication.
    The multiplication is split into two independent 1x3x7 problems:
    one for the first row of the output and one for the second row.
    Each 1x3x7 problem has rank 3.
    """
    n, m, p = DIMENSIONS
    rank = 6
    
    # U shape: (n*m, rank) = (6, 6)
    # V shape: (m*p, rank) = (21, 6)
    # W shape: (p*n, rank) = (14, 6)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Compute row 0 of C (i=0)
    # C[0, k] = A[0, 0]*B[0, k] + A[0, 1]*B[1, k] + A[0, 2]*B[2, k]
    
    # r=0: A[0,0] * B[0,k] -> C[0,k]
    U[0][0] = 1  # i=0, j=0
    for k in range(p):
        V[0 * p + k][0] = 1  # j=0, k
        W[k * n + 0][0] = 1  # k, i=0
        
    # r=1: A[0,1] * B[1,k] -> C[0,k]
    U[1][1] = 1  # i=0, j=1
    for k in range(p):
        V[1 * p + k][1] = 1  # j=1, k
        W[k * n + 0][1] = 1  # k, i=0
        
    # r=2: A[0,2] * B[2,k] -> C[0,k]
    U[2][2] = 1  # i=0, j=2
    for k in range(p):
        V[2 * p + k][2] = 1  # j=2, k
        W[k * n + 0][2] = 1  # k, i=0
        
    # Part 2: Compute row 1 of C (i=1)
    # C[1, k] = A[1, 0]*B[0, k] + A[1, 1]*B[1, k] + A[1, 2]*B[2, k]
    
    # r=3: A[1,0] * B[0,k] -> C[1,k]
    U[3][3] = 1  # i=1, j=0 (index 1*m + 0 = 3)
    for k in range(p):
        V[0 * p + k][3] = 1  # j=0, k
        W[k * n + 1][3] = 1  # k, i=1
        
    # r=4: A[1,1] * B[1,k] -> C[1,k]
    U[4][4] = 1  # i=1, j=1 (index 1*m + 1 = 4)
    for k in range(p):
        V[1 * p + k][4] = 1  # j=1, k
        W[k * n + 1][4] = 1  # k, i=1
        
    # r=5: A[1,2] * B[2,k] -> C[1,k]
    U[5][5] = 1  # i=1, j=2 (index 1*m + 2 = 5)
    for k in range(p):
        V[2 * p + k][5] = 1  # j=2, k
        W[k * n + 1][5] = 1  # k, i=1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
