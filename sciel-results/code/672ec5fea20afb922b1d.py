<<<<<<< SEARCH
def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]*B[j,k].
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c] should equal sum_j U[i*m+j,c]*V[j*p+k,c].
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,k): sum_j U3[i,j,:] * V3[j,k,:] should equal W3[k,i,:]
    # Wait, the tensor decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let's verify this condition directly as it's the definition of the rank.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
=======
def _validate(U, V, W, n, m, p):
    """Check exact decomposition using vectorized NumPy operations."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need S[i,j,k] == 1 for all i, j, k
    S = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(S, 1.0, atol=1e-9)
>>>>>>> REPLACE

