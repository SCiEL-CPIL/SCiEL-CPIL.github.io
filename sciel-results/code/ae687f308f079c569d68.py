"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Trivial decomposition: rank = m
    # For each j in 0..m-1:
    #   U[i*m+j, j] = 1 for all i
    #   V[j*p+k, j] = 1 for all k
    #   W[k*n+i, j] = 1 for all k, i
    # This gives sum_j 1*1*1 = m... wait, that gives m, not 1.
    
    # Actually, the tensor entry T[i,l,k] should be 1, and
    # sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] should equal 1.
    # With rank m and U[i*m+l, l] = 1, V[l*p+k, l] = 1, W[k*n+i, l] = 1:
    # sum_r ... = U[i*m+l, l] * V[l*p+k, l] * W[k*n+i, l] = 1*1*1 = 1 ✓
    # (only r=l contributes since other entries are 0)
    
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
