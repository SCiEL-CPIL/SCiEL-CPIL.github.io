"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    # Kronecker product decomposition: (3,1,1) x (1,5,5)
    # Rank = 3 * 5 = 15
    r1 = n  # rank of (3,1,1) tensor = 3
    r2 = m  # rank of (1,5,5) tensor = 5
    rank = r1 * r2  # 15
    
    U = [[0] * rank for _ in range(n * m)]  # (15, 15)
    V = [[0] * rank for _ in range(m * p)]  # (25, 15)
    W = [[0] * rank for _ in range(p * n)]  # (15, 15)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U[i*m + j, c] = 1 if i == c1 and j == c2
            U[c1 * m + c2][c] = 1
            # V[j*p + k, c] = 1 if j == c2 and k == c2
            V[c2 * p + c2][c] = 1
            # W[k*n + i, c] = 1 if k == c2 and i == c1
            W[c2 * n + c1][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
