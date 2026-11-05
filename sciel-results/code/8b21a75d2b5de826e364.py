import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Rank 1 decomposition of the all-ones tensor
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
