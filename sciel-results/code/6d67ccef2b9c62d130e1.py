"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook matrix multiplication tensor decomposition.
    For n x m * m x p, the rank is n * m * p.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) is valid, else 0.
    Actually, the tensor T is defined such that T[i*m+j, j*p+k, k*n+i] = 1.
    The schoolbook algorithm sets exactly one term in the sum to 1 for each (i,j,k)
    and 0 for all other index combinations in the tensor space, effectively
    implementing the standard multiplication.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m + j
                U[i * m + j][column] = 1
                # V index: row j*p + k
                V[j * p + k][column] = 1
                # W index: row k*n + i
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
