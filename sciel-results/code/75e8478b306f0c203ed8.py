DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split n=3 into n1=2 and n2=1
    n1, n2 = 2, 1
    
    # Rank for M_{n1, m, p} is n1*m*p (trivial)
    rank1 = n1 * m * p
    # Rank for M_{n2, m, p} is n2*m*p (trivial)
    rank2 = n2 * m * p
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: i in [0, n1-1]
    for i in range(n1):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: i in [n1, n-1]
    for i in range(n1, n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
