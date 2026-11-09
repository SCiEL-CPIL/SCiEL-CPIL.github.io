import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48 schoolbook
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Try to find a lower rank decomposition using known algorithms
    # For 3x4 * 4x4, we can try Strassen-like block decomposition
    # But first, let's try a numerical approach to see if we can do better
    
    # The schoolbook rank is 48. Known results suggest that for small matrices,
    # the rank might be reducible. Let's try to find a decomposition with rank < 48.
    
    # For now, return the schoolbook decomposition as a baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
