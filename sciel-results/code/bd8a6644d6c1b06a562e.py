"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for n×m times m×p is the all-ones tensor
    # of size n×m×p. The rank of the all-ones tensor J_{a,b,c} is min(a,b,c).
    # Here min(4,4,9) = 4, so rank = 4.
    # Decomposition: J[i,j,k] = sum_{c=0}^{3} delta_{i,c} * 1 * 1
    rank = min(n, m, p)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1 if i == c else 0
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
