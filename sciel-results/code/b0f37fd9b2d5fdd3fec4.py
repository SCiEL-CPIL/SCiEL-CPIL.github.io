"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 64
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Naive decomposition: C[i,j] = sum_k A[i,k] * B[k,j]
    # U has shape (n*m, rank) = (8, 64)
    # V has shape (m*p, rank) = (32, 64)
    # W has shape (p*n, rank) = (16, 64)
    
    # For each k in 0..m-1 and each i in 0..n-1 and each j in 0..p-1:
    # U[i*m + k, col] = 1
    # V[k*p + j, col] = 1
    # W[j*n + i, col] = 1
    
    col = 0
    for k in range(m):
        for i in range(n):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
