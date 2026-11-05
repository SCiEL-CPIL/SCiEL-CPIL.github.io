"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    # Rank = m * p = 4 * 8 = 32
    rank = m * p
    # U: (n*m, rank) = (12, 32)
    # V: (m*p, rank) = (32, 32)
    # W: (p*n, rank) = (24, 32)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in range(m):  # inner index value
        for k0 in range(p):  # output column index
            # U[i*m + j, col] = 1 if j == j0
            for i in range(n):
                U[i * m + j0][col] = 1
            # V[j'*p + k, col] = 1 if j' == j0 and k == k0
            V[j0 * p + k0][col] = 1
            # W[k'*n + i', col] = 1 if k' == k0
            for i_prime in range(n):
                W[k0 * n + i_prime][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
