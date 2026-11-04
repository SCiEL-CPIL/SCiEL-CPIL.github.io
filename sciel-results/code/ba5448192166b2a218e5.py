def construct():
    n, m, p = 2, 4, 4
    # Naive rank: n*m*p = 2*4*4 = 32
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Map r to indices i, j, k
        # i in 0..n-1, j in 0..p-1, k in 0..m-1
        # r = i * (p * m) + j * m + k
        k = r % m
        j = (r // m) % p
        i = r // (p * m)
        
        # U index: A[i, k] -> i * m + k
        u_idx = i * m + k
        # V index: B[k, j] -> k * p + j
        v_idx = k * p + j
        # W index: C[i, j] -> i * p + j
        w_idx = i * p + j
        
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
        
    return {"U": U, "V": V, "W": W}
