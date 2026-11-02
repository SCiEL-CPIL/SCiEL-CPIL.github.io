def _fast_residual(U, V, W, n, m, p, rank):
    """
    Compute residual R[i,j,k] = 1 - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    # Reshape
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    Total = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U_r[:, :, r] # (n, m)
        v = V_r[:, :, r] # (m, p)
        w = W_r[:, :, r] # (p, n)
        
        # P[i, j, k] = u[i, j] * v[j, k]
        # einsum 'ij,jk->ijk'
        P = np.einsum('ij,jk->ijk', u, v)
        
        # A[i, j, k] = P[i, j, k] * w[k, i]
        # w is (p, n). We need w[k, i] where k is axis 2, i is axis 0 of P.
        # w.T is (n, p). w.T[i, k].
        w_t = w.T # (n, p)
        w_b = w_t[:, np.newaxis, :] # (n, 1, p)
        
        Total += P * w_b
        
    return 1.0 - Total
