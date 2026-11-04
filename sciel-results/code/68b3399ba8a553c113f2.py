import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18
    rank = n * m * p
    U = np.zeros((n, rank))
    V = np.zeros((m, rank))
    W = np.zeros((p, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i, col] = 1
                V[j, col] = 1
                W[k, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
