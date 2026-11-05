"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    # We split the 6 columns of B into two groups of 3.
    # This reduces the problem to two independent 3x3x3 multiplications.
    # The rank of 3x3x3 schoolbook is 9.
    # Total rank = 9 + 9 = 18.
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for 3x3x3
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For the first block (columns 0-2 of B, 0-2 of C):
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns of B
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
                
    # For the second block (columns 3-5 of B, 3-5 of C):
    for i in range(n):
        for j in range(m):
            for k in range(3, 6): # Last 3 columns of B
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
