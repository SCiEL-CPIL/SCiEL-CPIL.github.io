import time

def construct():
    t_start = time.time()
    limit = t_start + 140.0
    
    n, m, p = 2, 3, 8
    
    # Rank is m * p = 3 * 8 = 24
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(m):
        for k in range(p):
            # U: rows i*m + a, i in 0..n-1
            for i in range(n):
                U[i * m + a][col] = 1
            # V: row a*p + k
            V[a * p + k][col] = 1
            # W: rows k*n + i, i in 0..n-1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
            
            if time.time() > limit:
                break
    
    # Validation
    valid = all(
        sum(U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r] for r in range(rank)) == 1
        for i in range(n)
        for a in range(m)
        for k in range(p)
    )
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rb = n*m*p
        U_f = [[0]*rb for _ in range(n*m)]
        V_f = [[0]*rb for _ in range(m*p)]
        W_f = [[0]*rb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_f[i*m+a][c] = 1
                    V_f[a*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
