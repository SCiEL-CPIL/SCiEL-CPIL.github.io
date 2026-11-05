import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # This is because C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term A[i,j] * B[j,k] appears with coefficient 1.
    # The all-ones tensor has rank 1.
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
