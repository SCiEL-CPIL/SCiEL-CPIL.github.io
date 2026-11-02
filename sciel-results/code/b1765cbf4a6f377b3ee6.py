"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * p
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The decomposition is sum over c (output index) of e_i otimes 1 otimes g_c
    # where c = k * n + i
    # U[i*m + j, c] = 1 if i = c % n
    # V[j*p + k, c] = 1 if k = c // n
    # W[c, c'] = 1 if c == c'
    
    for c in range(rank):
        i = c % n
        k = c // n
        for j in range(m):
            U[i * m + j][c] = 1
            V[j * p + k][c] = 1
        W[c][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
