"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np

# EVOLVE-BLOCK-START

def _build_recursive(n, m, p):
    """
    Recursively build a decomposition of M_{n,m,p} by splitting the first dimension.
    Base cases:
      n == 1: rank = p
      m == 1: rank = n * p
    Otherwise split n into n1, n2 and combine with m, p.
    """
    # Base cases
    if n == 1:
        # U: shape (m, p), V: shape (m*p, p), W: shape (p, p)
        U = np.zeros((m, p), dtype=int)
        V = np.zeros((m * p, p), dtype=int)
        W = np.eye(p, dtype=int)
        for j in range(m):
            for k in range(p):
                U[j, k] = 1
                V[j * p + k, k] = 1
        return U, V, W
    
    if m == 1:
        # U: shape (n, n*p), V: shape (p, n*p), W: shape (p*n, n*p)
        # For n x 1 * 1 x p, the result is n x p.
        # U[i, r] where r = i*p + k. U[i, i*p+k] = 1.
        # V[j*p + k, r] where j=0. V[k, i*p+k] = 1.
        # W[k*n + i, r] where r = i*p + k. W[k*n + i, i*p+k] = 1.
        rank = n * p
        U = np.zeros((n, rank), dtype=int)
        V = np.zeros((p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for k in range(p):
                r = i * p + k
                U[i, r] = 1
                V[k, r] = 1
                W[k * n + i, r] = 1
        return U, V, W
    
    # Recursive step: split n into n1, n2
    # Choose n1 = n // 2, n2 = n - n1
    n1 = n // 2
    n2 = n - n1
    
    # Get decompositions for (n1, m, p) and (n2, m, p)
    U1, V1, W1 = _build_recursive(n1, m, p)
    U2, V2, W2 = _build_recursive(n2, m, p)
    
    # Combine
    # U: (n*m, rank1 + rank2)
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Fill U1 block (rows 0 to n1*m - 1)
    U[:n1 * m, :rank1] = U1
    V[:, :rank1] = V1
    W[:p * n1, :rank1] = W1
    
    # Fill U2 block (rows n1*m to n*m - 1)
    U[n1 * m:, rank1:] = U2
    V[:, rank1:] = V2
    W[p * n1:, rank1:] = W2
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Compute tensor
    T_computed = np.zeros((n*m, m*p, p*n), dtype=int)
    for r in range(rank):
        T_computed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1
    
    return np.allclose(T_computed, T_target)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_recursive(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
