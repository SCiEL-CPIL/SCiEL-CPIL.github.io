import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 1 decomposition of the all-ones tensor
    # T_{i,j,k} = 1 for all i,j,k
    # U_{i,j,1} = 1, V_{j,k,1} = 1, W_{k,i,1} = 1
    rank = 1
    
    # U shape (n*m, rank)
    U = [[1] * rank for _ in range(n * m)]
    # V shape (m*p, rank)
    V = [[1] * rank for _ in range(m * p)]
    # W shape (p*n, rank)
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
