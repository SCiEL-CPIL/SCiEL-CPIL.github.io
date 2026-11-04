def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Rank-minimal decomposition for n x m x p matrix multiplication
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Each term A[i,j] B[j,k] is a rank-1 tensor.
    # Since the j-index is distinct for each term, the total rank is m.
    
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        j = r
        for i in range(n):
            # U[i*m + j][r] = 1
            U[i * m + j][r] = 1
            
            for k in range(p):
                # V[j*p + k][r] = 1
                V[j * p + k][r] = 1
                
                # W[k*n + i][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
