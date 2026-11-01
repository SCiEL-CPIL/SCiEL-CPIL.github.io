def construct():
    n, m, p = 2, 2, 8
    rank = p  # 8
    
    U = [[0]*rank for _ in range(n*m)]  # shape (4, 8)
    V = [[0]*rank for _ in range(m*p)]  # shape (16, 8)
    W = [[0]*rank for _ in range(p*n)]  # shape (16, 8)
    
    for r in range(rank):
        # U: all ones for this r
        for i in range(n):
            for j in range(m):
                U[i*m + j][r] = 1
        # V: 1 at column r for all rows j
        for j in range(m):
            V[j*p + r][r] = 1
        # W: 1 at row r for all columns i
        for i in range(n):
            W[r*n + i][r] = 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                assert abs(s - 1.0) < 1e-9, f"Failed at i={i},j={j},k={k}: s={s}"
    
    return {"U": U, "V": V, "W": W}
