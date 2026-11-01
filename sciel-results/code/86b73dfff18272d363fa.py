import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i = r // m
        j_prime = r % m
        
        # U[i * m + j, r] = delta_{j, j_prime}
        for j in range(m):
            if j == j_prime:
                U[i * m + j][r] = 1
        
        # W[k * n + i, r] = 1
        for k in range(p):
            W[k * n + i][r] = 1
            
        # V[j * p + k, r] = delta_{j, j_prime}
        for j in range(m):
            if j == j_prime:
                for k in range(p):
                    V[j * p + k][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
