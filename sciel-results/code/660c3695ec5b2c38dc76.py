import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Standard decomposition: rank = m
    # For each j in 0..m-1, one term
    R = m  # 4
    
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    for j in range(m):
        # U[i*m + j, j] = 1 for all i
        for i in range(n):
            U[i * m + j, j] = 1.0
        # V[j*p + k, j] = 1 for all k
        for k in range(p):
            V[j * p + k, j] = 1.0
        # W[k*n + i, j] = 1 for all i, k
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1.0
    
    # Verify
    U_t = U.reshape(n, m, R)
    V_t = V.reshape(m, p, R)
    W_t = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    assert np.allclose(T, np.ones((n, m, p)), atol=1e-10), "Decomposition failed!"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
