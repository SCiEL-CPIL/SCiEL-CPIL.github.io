    def _loss_only(params, n, m, p, r):
        size_U = n * m * r
        size_V = m * p * r
        U = params[:size_U].reshape(n * m, r)
        V = params[size_U:size_U + size_V].reshape(m * p, r)
        W = params[size_U + size_V:].reshape(p * n, r)
        
        T = np.zeros((n, m, p))
        # Vectorized over c?
        # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We can precompute indices
        # Or just loop c, which is 16 iterations.
        for c in range(r):
            Uc = U[:, c]
            Vc = V[:, c]
            Wc = W[:, c]
            # Uc is 6x1, Vc is 9x1, Wc is 6x1
            # We need to sum Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
            # This is a specific contraction.
            # For n=2, m=3, p=3, we can just do the 18 multiplies.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
                        
        err = 1.0 - T
        return np.sum(err**2)
    