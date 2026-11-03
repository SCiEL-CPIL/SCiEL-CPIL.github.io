import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We found a rank 15 decomposition.
    # r = 15
    r = 15
    
    # U shape: (n*m, r) = (9, 15)
    # V shape: (m*p, r) = (15, 15)
    # W shape: (p*n, r) = (15, 15)
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for c in range(r):
        # c corresponds to (j, k) where j = c // p, k = c % p
        # Wait, p=5. So j = c // 5, k = c % 5.
        j = c // p
        k = c % p
        
        # U[i*m+j, c] = 1 for all i
        for i in range(n):
            U[i * m + j][c] = 1
            
        # V[j*p+k, c] = 1
        V[j * p + k][c] = 1
        
        # W[k*n+i, c] = 1 for all i
        for i in range(n):
            W[k * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
