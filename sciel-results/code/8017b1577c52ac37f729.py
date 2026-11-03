"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p for 3x5x5 matrix multiplication.
    
    The matrix multiplication tensor T_{i,j,k} = 1 for all valid (i,j,k) where:
    - i in [0, n) is the row of A (and C)
    - j in [0, m) is the column of A / row of B  
    - k in [0, p) is the column of B (and C)
    
    The bilinear decomposition requires:
    C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook algorithm, each (i,j,k) triple gets one rank term:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    This gives rank = n*m*p = 3*5*5 = 75.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
