# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses the Kronecker product structure M_{2,5,5} = M_{2,1,1} \otimes M_{1,5,5}.
    Rank(M_{2,1,1}) = 2.
    Rank(M_{1,5,5}) = 25.
    Total Rank = 2 * 25 = 50.
    """
    n, m, p = 2, 5, 5
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterating over the Kronecker factors
    # Factor 1: 2x1 * 1x1 (indices i, j=0, k=0) -> Rank 2
    # Factor 2: 1x5 * 5x5 (indices i=0, j, k) -> Rank 25
    # Combined: i in 0..1, j in 0..4, k in 0..4
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
