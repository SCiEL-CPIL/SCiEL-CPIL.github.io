import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # is defined by T[i, j, k] = 1 for the specific index mapping.
    # The standard schoolbook algorithm uses n*m*p terms.
    # Each term r corresponds to a unique (i, j, k) triplet.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The index for U is i*m + j.
    # The index for V is j*p + k.
    # The index for W is k*n + i.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
