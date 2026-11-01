import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 4
    # Target: find rank-minimal decomposition for 2x3 * 3x4
    # Trivial rank is 24. Try to find lower rank.
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from 16 upward
    for rank in range(16, 25):
        # Random search for decomposition
        found = False
        for trial in range(200):
            np.random.seed(trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Check if sum_r U[:,r] * V[:,r] * W[:,r] == T
            # i.e., for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-8):
                found = True
                break
        
        if found:
            # Try to simplify to integer/half-integer
            # For now, just return the found solution
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to trivial
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
