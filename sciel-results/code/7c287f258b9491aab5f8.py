"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            r = j * n + i
            # U index: i*m + j
            a = i * m + j
            U[a][r] = 1.0
            
            # V index: j*p + k
            for k in range(p):
                b = j * p + k
                V[b][r] = 1.0
                
            # W index: k*n + i
            for k in range(p):
                c = k * n + i
                W[c][r] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
