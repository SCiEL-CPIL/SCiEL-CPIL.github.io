import numpy as np


def construct():
    n, m, p = 4, 4, 5
    rank = 1
    
    # Rank-1 decomposition of the all-ones tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
