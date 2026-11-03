import time
def construct():
    deadline = time.time() + 140
    
    n, m, p = 4, 4, 5
    
    # Schoolbook decomposition
    # Rank = n * m * p = 4 * 4 * 5 = 80
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, r] = 1
                U[i*m + j][r] = 1.0
                # V[j*p + k, r] = 1
                V[j*p + k][r] = 1.0
                # W[k*n + i, r] = 1
                W[k*n + i][r] = 1.0
                r += 1
                
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at {i},{j},{k}: sum={s}")
                    
    return {"U": U, "V": V, "W": W}
