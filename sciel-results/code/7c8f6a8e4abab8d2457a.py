"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Schoolbook decomposition
    # Rank = n * m * p = 2 * 2 * 14 = 56
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all terms A[i][j] * B[j][k] -> C[i][k]
    # i in 0..n-1 (2)
    # j in 0..m-1 (2)
    # k in 0..p-1 (14)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                u_idx = i * m + j
                # V index: j * p + k
                v_idx = j * p + k
                # W index: k * n + i (Trace convention W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
