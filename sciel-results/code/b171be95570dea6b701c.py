import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 1 decomposition of the All-ones tensor
    rank = 1
    U = np.ones((nm, rank))
    V = np.ones((mp, rank))
    W = np.ones((pn, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
