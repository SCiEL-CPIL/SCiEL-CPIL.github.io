"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Reformatted schoolbook decomposition.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Generate indices for each rank-1 term
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
                
    for col, (i, j, k) in enumerate(indices):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
