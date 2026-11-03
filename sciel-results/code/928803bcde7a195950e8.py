def _objective_vectorized(params, n, m, p, r):
    n_eq = n * m * p
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    
    # We need to compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
    # This is a tensor contraction.
    # We can reshape U, V, W to use einsum or matmul tricks.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Note indices: U is (i*m+j), V is (j*p+k), W is (k*n+i).
    
    # Let's flatten the target tensor indices.
    # i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # We can compute this by iterating over c and accumulating? No, that's slow.
    # We can use np.einsum('ic,jc,kc->ijk', U_reshaped, V_reshaped, W_reshaped)
    # But U is (n*m, r). We need to reshape to (n, m, r).
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # einsum: 'ijc,jkc,kic->ijk'
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Target is 1.0 everywhere.
    residuals = T_computed - 1.0
    return residuals.flatten()
