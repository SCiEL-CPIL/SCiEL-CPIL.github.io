def _verify(U, V, W, n, m, p, rank):
    """Verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    # Reshape to tensors for vectorized multiplication
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # We can do this by summing over the last axis after broadcasting
    # Term = U_t[:, :, None, :] * V_t[None, :, :, :] * W_t[:, None, None, :] ? No, indices don't match directly for einsum without care.
    
    # Use einsum: 'ijc,jkc,kic->ijk'
    # U is indexed by i, j, c
    # V is indexed by j, k, c
    # W is indexed by k, i, c
    
    # Let's check dimensions:
    # U_t: (n, m, rank)
    # V_t: (m, p, rank)
    # W_t: (p, n, rank)
    
    # We want result[i, j, k]
    # Sum over c of U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
    
    # einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    # Check if all entries are 1
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=1e-9)
