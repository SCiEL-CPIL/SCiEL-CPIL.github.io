import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the schoolbook decomposition of matrix multiplication.
    Dimensions: n x m times m x p.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Rank is n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the interface example, but numpy is allowed.
    # The prompt says "factor matrices of shapes...". It doesn't strictly enforce list of lists vs numpy arrays,
    # but the seed program uses lists of lists. I will stick to lists of lists to be safe with the interface expectations
    # or convert to numpy if that's preferred. The seed returns lists.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
