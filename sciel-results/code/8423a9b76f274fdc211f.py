import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 5
    
    # Dimensions for factor matrices
    dim_U = n * m  # 16
    dim_V = m * p  # 20
    dim_W = p * n  # 20
    
    # Schoolbook rank
    rank = n * m * p # 80
    
    U = np.zeros((dim_U, rank), dtype=int)
    V = np.zeros((dim_V, rank), dtype=int)
    W = np.zeros((dim_W, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i*m + j, idx] = 1
                # V index: j*p + k
                V[j*p + k, idx] = 1
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k*n + i, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
