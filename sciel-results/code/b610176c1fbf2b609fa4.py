"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i_prime in range(n):
        for k_prime in range(p):
            r = i_prime * p + k_prime
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if i == i_prime else 0
            for j in range(m):
                for k in range(p):
                    V[j * p + k][r] = 1 if k == k_prime else 0
            for k in range(p):
                for i in range(n):
                    W[k * n + i][r] = 1 if (k == k_prime and i == i_prime) else 0
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
