"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: k=0, j in {0,1}
    for i in range(n):
        for j in range(2):
            U[i * m + j][0] = 1
    for j in range(2):
        V[j * p + 0][0] = 1
    for i in range(n):
        W[0 * n + i][0] = 1
        
    # Term 1: k=1, j in {0,1}
    for i in range(n):
        for j in range(2):
            U[i * m + j][1] = 1
    for j in range(2):
        V[j * p + 1][1] = 1
    for i in range(n):
        W[1 * n + i][1] = 1
        
    # Term 2: k=2, j in {0,1}
    for i in range(n):
        for j in range(2):
            U[i * m + j][2] = 1
    for j in range(2):
        V[j * p + 2][2] = 1
    for i in range(n):
        W[2 * n + i][2] = 1
        
    # Term 3: j=2, all k, all i
    for i in range(n):
        U[i * m + 2][3] = 1
    for k in range(p):
        V[2 * p + k][3] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][3] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
