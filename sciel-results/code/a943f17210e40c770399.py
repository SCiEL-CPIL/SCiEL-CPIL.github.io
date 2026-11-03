def construct():
    import time
    
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    n, m, p = 2, 3, 10
    # Trivial rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the trivial decomposition
    # The tensor T_{(i,j), (j',k'), (k'',i'')} is 1 if j=j', k=k'', i=i''
    # The rank-r term contributes U_{i,j,r} * V_{j',k',r} * W_{k'',i'',r}
    # We want this product to be 1 iff j=j', k=k'', i=i'' for the specific r associated with (i,j,k)
    # Let r index the triple (i,j,k).
    # U[i*m+j][r] = 1
    # V[j*p+k][r] = 1
    # W[k*n+i][r] = 1
    
    # Mapping r to (i, j, k)
    # r = i * (m * p) + j * p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                
                # U index: i * m + j
                U[i * m + j][r] = 1
                
                # V index: j * p + k
                V[j * p + k][r] = 1
                
                # W index: k * n + i
                W[k * n + i][r] = 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
