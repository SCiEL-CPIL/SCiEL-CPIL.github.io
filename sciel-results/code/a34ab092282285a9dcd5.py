import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check the 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check the 0s
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                support.add((u, v, w))
                
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) in support:
                    continue
                val = sum(U[u, c] * V[v, c] * W[w, c] for c in range(rank))
                if abs(val) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for n x m times m x p is the all-ones tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # A rank-1 decomposition: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # gives sum_c U*V*W = 1 for all (i,j,k), which is exactly what we need.
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    assert _verify(U, V, W, n, m, p), "Rank-1 decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
