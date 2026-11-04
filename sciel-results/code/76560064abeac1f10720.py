"""Matrix multiplication decomposition for 2x5 * 5x6 using schoolbook method."""

def construct():
    n, m, p = 2, 5, 6
    
    # U: shape (n*m, rank) = (10, rank), indexed by i*m + j
    # V: shape (m*p, rank) = (30, rank), indexed by j*p + k
    # W: shape (p*n, rank) = (12, rank), indexed by k*n + i (trace convention)
    
    rank = n * m * p  # schoolbook: 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
