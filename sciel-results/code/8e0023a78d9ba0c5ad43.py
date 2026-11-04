"""Matrix multiplication 2x5x6 - rank-minimal decomposition via unit-column splitting."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = m (split middle dimension into m unit columns)
    # Each 2x1xp block has rank 1 (outer product)
    rank = m
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for j in range(m):
        # U: rows are (i, j) pairs, index i*m + j
        for i in range(n):
            U[i * m + j, j] = 1.0
        # V: rows are (j, k) pairs, index j*p + k
        for k in range(p):
            V[j * p + k, j] = 1.0
        # W: rows are (k, i) pairs, index k*n + i
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1.0
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
