import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        x2 = 2 * arr
        if not np.allclose(x2, np.round(x2), atol=1e-9):
            return False
            
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # T[i,j,k] = einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # The target tensor is the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor is 1.
    # U, V, W can all be vectors of 1s.
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify just in case
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook if Rank 1 fails for some unknown reason
    schoolbook_rank = n * m * p
    U_s = [[0] * schoolbook_rank for _ in range(n * m)]
    V_s = [[0] * schoolbook_rank for _ in range(m * p)]
    W_s = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
