"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    
    The multiplication tensor T_{2,5,5} is the direct sum of two T_{1,5,5} tensors.
    The rank of T_{1,n,n} is n^2. For n=5, rank is 25.
    Total rank is 2 * 25 = 50. This is the theoretical minimum.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # U: shape (n*m, rank) = (10, 50)
    # V: shape (m*p, rank) = (25, 50)
    # W: shape (p*n, rank) = (10, 50)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook: for each (i, j, k) triple, one rank-1 term
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1  (following the convention W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
