import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Sum = 1*1*1 = 1
    
    R = 1
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
