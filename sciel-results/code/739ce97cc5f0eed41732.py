import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Use float to be safe with "integer/half-integer"
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # ... same logic ...
    
    return {"U": U, "V": V, "W": W}
