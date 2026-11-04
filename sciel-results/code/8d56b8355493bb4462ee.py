def construct():
    n, m, p = 3, 3, 8
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Split B into 4 blocks of width 2
    for block_idx in range(4):
        start_k = block_idx * 2
        # For each of the 3 rows of B (k in 0..2)
        for k in range(m):
            # M_r(A) = sum_{i=0}^{2} A[i][k]
            # U index: i*m + k
            for i in range(n):
                U[i * m + k][col] = 1
            
            # N_r(B) = B[k][start_k]
            # V index: k*p + (start_k)
            V[k * p + start_k][col] = 1
            
            # W: for each i, C[i][start_k] gets contribution
            # W index: (start_k)*n + i
            for i in range(n):
                W[(start_k) * n + i][col] = 1
            
            col += 1
            
            # Second column of the block
            # M_r(A) = sum_{i=0}^{2} A[i][k]
            for i in range(n):
                U[i * m + k][col] = 1
            
            # N_r(B) = B[k][start_k + 1]
            V[k * p + start_k + 1][col] = 1
            
            # W: for each i, C[i][start_k+1] gets contribution
            for i in range(n):
                W[(start_k + 1) * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
