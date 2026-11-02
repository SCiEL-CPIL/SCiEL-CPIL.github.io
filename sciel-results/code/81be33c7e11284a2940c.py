"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x5 by 5x5.
    Uses the direct sum structure: M_{2,5,5} = M_{1,5,5} \oplus M_{1,5,5}.
    The rank is 2 * (5 * 5) = 50.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
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
