"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130
    
    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    r_school = n * m * p
    U = np.zeros((n*m, r_school))
    V = np.zeros((m*p, r_school))
    W = np.zeros((p*n, r_school))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    
    best_U, best_V, best_W = U, V, W
    
    # Attempt lower-rank decomposition via ALS
    for target_r in [24, 28, 30, 32, 34]:
        if time.time() > deadline - 20:
            break
        result = _try_als(T, n, m, p, target_r, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _try_als(T, n, m, p, r, deadline):
    a, b, c = n*m, m*p, p*n
    for seed in range(2):
        if time.time() > deadline - 15:
            return None
        rng = np.random.RandomState(seed + 100)
        U = rng.randn(a, r) * 0.1
        V = rng.randn(b, r) * 0.1
        W = rng.randn(c, r) * 0.1
        
        for it in range(80):
            if time.time() > deadline - 15:
                return None
            for l in range(r):
                # Compute residual excluding component l
                R = T.copy()
                for l2 in range(r):
                    if l2 != l:
                        R -= np.outer(U[:, l2], np.outer(V[:, l2], W[:, l2]))
                
                # Update U[:,l]
                vw = np.outer(V[:, l], W[:, l]).ravel()
                d = np.dot(vw, vw)
                if d > 1e-14:
                    U[:, l] = (R.reshape(a, -1) @ vw) / d
                
                # Update V[:,l]
                uw = np.outer(U[:, l], W[:, l]).ravel()
                d = np.dot(uw, uw)
                if d > 1e-14:
                    V[:, l] = (R.transpose(1, 0, 2).reshape(b, -1) @ uw) / d
                
                # Update W[:,l]
                uv = np.outer(U[:, l], V[:, l]).ravel()
                d = np.dot(uv, uv)
                if d > 1e-14:
                    W[:, l] = (R.transpose(2, 0, 1).reshape(c, -1) @ uv) / d
            
            # Check convergence
            Trec = np.zeros_like(T)
            for l in range(r):
                Trec += np.outer(U[:, l], np.outer(V[:, l], W[:, l]))
            if np.linalg.norm(T - Trec) < 1e-9:
                return U, V, W
    return None
# EVOLVE-BLOCK-END
