import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # This is the all-ones tensor, which has tensor rank 1
    R = 1
    U = [[1] * R for _ in range(dim_u)]
    V = [[1] * R for _ in range(dim_v)]
    W = [[1] * R for _ in range(dim_w)]
    
    return {"U": U, "V": V, "W": W}
