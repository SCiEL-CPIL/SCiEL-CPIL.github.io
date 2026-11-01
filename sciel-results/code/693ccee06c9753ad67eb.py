import numpy as np

def construct():
    # ... existing logic ...
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # ... fill logic ...
    
    return {"U": U, "V": V, "W": W}
