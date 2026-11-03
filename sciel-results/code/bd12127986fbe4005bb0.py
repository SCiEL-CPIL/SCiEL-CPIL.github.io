"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Map r to (i, j, k)
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
