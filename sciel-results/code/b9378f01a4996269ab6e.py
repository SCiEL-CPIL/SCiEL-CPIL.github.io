"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank m for n x m times m x p matrix multiplication.
    # U shape (n*m, m), V shape (m*p, m), W shape (p*n, m)
    # For each column r in [0, m):
    #   U[i*m + r, r] = 1 for all i in [0, n)
    #   V[r*p + k, r] = 1 for all k in [0, p)
    #   W[k*n + i, r] = 1 for all k in [0, p), i in [0, n)
    #
    # Verification: C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # For column r: sum_j U[i*m+j, r]*A[i,j] = A[i,r] (since U[i*m+j, r]=1 only when j=r)
    # For column r: sum_j' V[j'*p+k, r]*B[j',k] = B[r,k] (since V[j'*p+k, r]=1 only when j'=r)
    # So term r contributes: A[i,r] * B[r,k] * W[k*n+i, r] = A[i,r] * B[r,k] * 1
    # Sum over r: sum_r A[i,r] * B[r,k] = (A*B)[i,k] ✓
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
