"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r corresponds to (alpha, j_prime)
    # alpha in 0..n-1, j_prime in 0..m-1
    col = 0
    for alpha in range(n):
        for j_prime in range(m):
            # U[i, j, r] = delta(i, alpha) * delta(j, j_prime)
            for i in range(n):
                for j in range(m):
                    if i == alpha and j == j_prime:
                        U[i * m + j][col] = 1
            # V[j, k, r] = delta(j, j_prime)
            # Note: V index is j*p + k
            for j in range(m):
                for k in range(p):
                    if j == j_prime:
                        V[j * p + k][col] = 1
            # W[k, i, r] = delta(i, alpha)
            # Note: W index is k*n + i
            for k in range(p):
                for i in range(n):
                    if i == alpha:
                        W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
