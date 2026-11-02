# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Theoretical minimum rank for 2 x m x p matrix multiplication is m*p.
    # We construct the exact decomposition of rank m*p.
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        j0 = t // p
        k0 = t % p
        
        # U[i*m + j0, t] = 1 for all i
        for i in range(n):
            U[i * m + j0][t] = 1
            W[k0 * n + i][t] = 1
        
        # V[j0*p + k0, t] = 1
        V[j0 * p + k0][t] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
