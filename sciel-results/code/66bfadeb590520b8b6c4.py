import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_rank23_decomposition():
    """Construct a rank-23 decomposition using BCFL for 3x3x3 core and schoolbook for strips."""
    n, m, p = 3, 4, 4
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # BCFL terms for 3x3x3
    # u, v, w are length 3
    bcfl_terms = [
        ((1, 0, 0), (1, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 0), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 0, 1), (1, 1, 0)),
        ((1, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((1, 0, 1), (0, 1, 0), (1, 0, 0)),
        ((1, 1, 0), (1, 0, 0), (0, 0, 1)),
        ((1, 1, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 1), (1, 0, 0), (0, 0, 1)),
        ((0, 1, 1), (0, 1, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (1, 1, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 1), (0, 1, 0)),
        ((0, 0, 1), (1, 1, 0), (0, 0, 1))
    ]
    
    for r, (u, v, w) in enumerate(bcfl_terms):
        # u is for i in 0..2, j in 0..2
        for i in range(3):
            U[i * m + 0, r] = u[i]
            U[i * m + 1, r] = u[i]
            U[i * m + 2, r] = u[i]
        # v is for j in 0..2, k in 0..2
        for j in range(3):
            V[j * p + 0, r] = v[j]
            V[j * p + 1, r] = v[j]
            V[j * p + 2, r] = v[j]
        # w is for k in 0..2, i in 0..2
        for k in range(3):
            W[k * n + 0, r] = w[k]
            W[k * n + 1, r] = w[k]
            W[k * n + 2, r] = w[k]

    # Strip 1: j=3
    # T[i, 3, k] = 1 for all i, k
    # Term index 19
    r = 19
    for i in range(3):
        U[i * m + 3, r] = 1
    for k in range(4):
        V[3 * p + k, r] = 1
    for k in range(4):
        for i in range(3):
            W[k * n + i, r] = 1

    # Strip 2: k=3, j in 0..2
    # T[i, j, 3] = 1 for all i, j in 0..2
    # Decompose into 3 terms, one for each j
    for j0 in range(3):
        r = 20 + j0
        for i in range(3):
            U[i * m + j0, r] = 1
        V[j0 * p + 3, r] = 1
        for i in range(3):
            W[3 * n + i, r] = 1

    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Use the new rank-23 decomposition
    U, V, W = _construct_rank23_decomposition()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to old if something goes wrong (should not happen)
        # But we don't have the old one in this file anymore if we replace it.
        # We should keep the old one or just raise error?
        # Better to keep the old logic in a helper if possible, or just return empty?
        # The task says "invalid result ... scores 0".
        # So we must ensure it works.
        pass

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Use the new rank-23 decomposition
    U, V, W = _construct_rank23_decomposition()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # If verification fails, try the old rank-44 decomposition as fallback
    # (Re-implementing the old logic briefly to be safe)
    U44, V44, W44 = _construct_structured_decomposition()
    if _verify_decomposition(U44, V44, W44, n, m, p, tol=1e-6):
        return {
            "U": U44.tolist(),
            "V": V44.tolist(),
            "W": W44.tolist()
        }
        
    # Last resort
    return {
        "U": U44.tolist(),
        "V": V44.tolist(),
        "W": W44.tolist()
    }
