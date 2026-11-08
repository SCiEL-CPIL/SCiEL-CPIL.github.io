<<<<<<< SEARCH
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}
=======
    # Verify
    if verify(U, V, W):
        # Attempt rank reduction
        U, V, W, reduced_rank = reduce_rank_gauss(U, V, W)
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        # If reduction broke validity, use original
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

Wait, the reduce_rank_gauss function as written has a bug - after breaking out, it returns the potentially broken U,V,W. Let me fix this properly.

