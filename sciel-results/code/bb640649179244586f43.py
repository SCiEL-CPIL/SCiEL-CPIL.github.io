"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication
    of dimensions n x m and m x p.
    The rank is n * m * p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to A[i, j]
                U[i * m + j][column] = 1
                # V corresponds to B[j, k]
                V[j * p + k][column] = 1
                # W corresponds to C[k, i] based on trace convention W[k*n+i, r]
                # The task specifies: "output factor follows the official trace convention W[k*n+i,r]"
                # Standard multiplication C[i, k] = sum_j A[i, j] B[j, k].
                # The tensor decomposition usually maps to C[i, k].
                # If W is indexed by k*n+i, it implies C is stored as C[k, i] or similar?
                # Let's check the seed code: W[k * n + i][column] = 1.
                # This matches the seed's convention.
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
