"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all triples (i, j, k)
    for idx in range(rank):
        # Map linear index to (i, j, k)
        # Order: i * m * p + j * p + k
        k = idx % p
        tmp = idx // p
        j = tmp % m
        i = tmp // m
        
        col = idx
        
        # U index: i * m + j
        u_idx = i * m + j
        U[u_idx][col] = 1
        
        # V index: j * p + k
        v_idx = j * p + k
        V[v_idx][col] = 1
        
        # W index: k * n + i
        w_idx = k * n + i
        W[w_idx][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
