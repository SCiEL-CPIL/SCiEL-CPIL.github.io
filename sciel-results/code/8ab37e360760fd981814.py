"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank = max(n * m, m * p, p * n)  # 15
    
    # U shape: (n*m, rank) = (9, 15)
    # V shape: (m*p, rank) = (15, 15)
    # W shape: (p*n, rank) = (15, 15)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Structured rank-15 decomposition:
    # Each rank-1 term c is indexed by (j, k) where j in [0,m), k in [0,p)
    # c = j * p + k
    # U[i*m + j, c] = 1 for all i, j, k  (i.e., U[i*m+j, j*p+k] = 1)
    # V[j*p + k, c] = 1 if c == j*p+k else 0  (identity)
    # W[k*n + i, c] = 1 for all i, j, k  (i.e., W[k*n+i, j*p+k] = 1)
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # V: identity structure
            V[j * p + k][c] = 1
            # U: for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # W: for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
