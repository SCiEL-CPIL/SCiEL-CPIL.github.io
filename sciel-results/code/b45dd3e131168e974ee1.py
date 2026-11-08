import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # ... (same logic, but using int 1, -1 instead of 1.0, -1.0)
    
    return {"U": np.array(U, dtype=float), "V": np.array(V, dtype=float), "W": np.array(W, dtype=float)}
