"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Iterate over rank indices to populate the factor matrices
    for r in range(rank):
        # Determine i, j, k from r
        # r = i * (m * p) + j * p + k
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx][r] = 1.0
        V[v_idx][r] = 1.0
        W[w_idx][r] = 1.0
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
