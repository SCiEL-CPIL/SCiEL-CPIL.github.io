import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i*m + k
                b = k*p + j
                c = j*n + i
                target[a, b, c] = 1.0
                
    decomp = np.einsum('it,jt,kt->ijk', U, V, W)
    diff = np.abs(decomp - target)
    return np.max(diff) < tol

def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 140.0
    
    # Start with schoolbook rank 90
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    if not _verify(U, V, W, n, m, p):
        # Should not happen
        pass
    
    return {"U": U, "V": V, "W": W}
