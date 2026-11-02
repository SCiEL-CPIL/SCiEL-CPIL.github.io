DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use rank = n * p construction
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i0 in range(n):
        for k0 in range(p):
            # U[i*m+j, col] = 1 if i == i0, else 0 (for all j)
            for i in range(n):
                for j in range(m):
                    if i == i0:
                        U[i * m + j][col] = 1
            # V[j*p+k, col] = 1 if k == k0, else 0 (for all j)
            for j in range(m):
                for k in range(p):
                    if k == k0:
                        V[j * p + k][col] = 1
            # W[k*n+i, col] = 1 if i == i0 and k == k0, else 0
            for k in range(p):
                for i in range(n):
                    if i == i0 and k == k0:
                        W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
