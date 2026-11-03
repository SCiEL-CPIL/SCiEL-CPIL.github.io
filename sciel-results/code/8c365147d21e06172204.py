DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block for i=1 (rows 0 in 0-based)
    # r = 0..4
    for r in range(5):
        for j in range(m):
            if j == r:
                U[0 * m + j][r] = 1
        for jp in range(m * p):
            j_prime = jp // p
            k = jp % p
            if j_prime == r:
                V[jp][r] = 1
        for k in range(p):
            W[k * n + 0][r] = 1
            
    # Block for i=2 (rows 1 in 0-based)
    # r = 5..9
    for r in range(5):
        for j in range(m):
            if j == r:
                U[1 * m + j][r + 5] = 1
        for jp in range(m * p):
            j_prime = jp // p
            k = jp % p
            if j_prime == r:
                V[jp][r + 5] = 1
        for k in range(p):
            W[k * n + 1][r + 5] = 1
            
    # Block for i=3 (rows 2 in 0-based)
    # r = 10..14
    for r in range(5):
        for j in range(m):
            if j == r:
                U[2 * m + j][r + 10] = 1
        for jp in range(m * p):
            j_prime = jp // p
            k = jp % p
            if j_prime == r:
                V[jp][r + 10] = 1
        for k in range(p):
            W[k * n + 2][r + 10] = 1
            
    return {"U": U, "V": V, "W": W}
