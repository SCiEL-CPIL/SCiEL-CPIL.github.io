DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank) -> Note: task says shape (p*n, rank) and index W[k*n+i, r]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j -> i*m + j
                idx_u = i * m + j
                # Index in V: row j, col k -> j*p + k
                idx_v = j * p + k
                # Index in W: output (i, k). Convention W[k*n + i, r]
                idx_w = k * n + i
                
                U[idx_u][r] = 1
                V[idx_v][r] = 1
                W[idx_w][r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
