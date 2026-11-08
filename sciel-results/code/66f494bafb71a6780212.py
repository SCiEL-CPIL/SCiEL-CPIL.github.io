def construct():
    import time
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    n, m, p = 4, 4, 5
    
    # Use schoolbook multiplication for 4x4 * 4x5
    # Rank = n * m * p = 4 * 4 * 5 = 80
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # Tensor term for (i, j, k):
    # U[i*m + j] = 1
    # V[j*p + k] = 1
    # W[k*n + i] = 1
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    # Verify quickly if possible (optional, but good for sanity)
    # Since it's schoolbook, it's guaranteed correct.
    
    return {"U": U, "V": V, "W": W}
