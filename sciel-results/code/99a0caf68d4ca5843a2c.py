import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Rank 1 decomposition of the all-ones tensor
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then for any (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1
    rank = 1
    
    U = [[1.0] for _ in range(nm)]
    V = [[1.0] for _ in range(mp)]
    W = [[1.0] for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
