import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition
    # ... (keep the U2, V2, W2 definition)
    
    # ... (keep the A_specs, B_specs, C_specs)

    # Instead of the loop, we can generate the columns directly.
    # But the loop is fine.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
