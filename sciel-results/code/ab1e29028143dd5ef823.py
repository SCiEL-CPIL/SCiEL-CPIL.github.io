import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col_idx] = 1
                V[j * p + k, col_idx] = 1
                W[k * n + i, col_idx] = 1
                col_idx += 1
                
    # Convert to lists if required by interface, but numpy arrays are usually accepted
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't strictly say lists, but the seed used lists.
    # To be safe and "new", I'll return numpy arrays.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
