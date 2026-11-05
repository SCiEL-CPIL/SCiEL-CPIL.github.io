import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def _clean_entry(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    doubled = v * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return round(doubled) / 2.0
    return v

def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank_36():
    # Same as before
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        # ... standard strassen
        pass

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        # ... standard schoolbook
        pass
        
    # ... construction logic
    
    return U, V, W

def construct():
    start_time = time.time()
    # Try to find rank 9 for 2x2x3
    # If found, build rank 32
    # Else build rank 36
    pass
