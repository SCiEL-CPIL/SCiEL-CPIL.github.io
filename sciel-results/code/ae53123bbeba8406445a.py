import numpy as np
from scipy.optimize import lsq_linear
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try to find minimal rank decomposition
    # Lower bound: max(nm, mp, pn)
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, nm * mp * pn + 1):
        result = try_rank(rank, n, m, p, nm, mp, pn)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook(n, m, p, nm, mp, pn)

def schoolbook(n, m, p, nm, mp, pn):
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank(rank, n, m, p, nm, mp, pn):
    # This is a placeholder; actual implementation would search for valid U,V,W
    # For now, return None to indicate failure
    return None
