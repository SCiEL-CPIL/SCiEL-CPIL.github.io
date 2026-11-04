"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use the merge strategy over k for each (i, j).
    # This reduces the rank from n*m*p to n*m.
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i = r // m
        j = r % m
        
        # U: 1 at (i, j)
        U[i * m + j][r] = 1
        
        # V: 1 at (j, k) for all k
        for k in range(p):
            V[j * p + k][r] = 1
            
        # W: 1 at (i, k) for all k, stored as k*n + i
        for k in range(p):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
