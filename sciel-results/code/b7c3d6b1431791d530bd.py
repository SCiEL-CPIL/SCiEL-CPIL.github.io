def construct():
    n, m, p = 3, 3, 5
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Enumerate (i,k) pairs and assign them to rank indices
    idx = 0
    for i in range(n):
        for k in range(p):
            # U[i*m+j, idx] = 1 for all j
            for j in range(m):
                U[i * m + j][idx] = 1
            # V[j*p+k, idx] = 1 for all j
            for j in range(m):
                V[j * p + k][idx] = 1
            # W[k*n+i, idx] = 1 for all i
            for ii in range(n):
                W[k * n + ii][idx] = 1
            idx += 1
    
    return {"U": U, "V": V, "W": W}
