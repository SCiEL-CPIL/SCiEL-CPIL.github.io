def _try_rank_reduction(U71, V71, W71, n, m, p, target_rank=70, time_limit=10.0):
    """
    Attempt to reduce the rank of the given decomposition to target_rank.
    """
    R = target_rank
    # Extract first R columns
    U_init = U71[:, :R].copy()
    V_init = V71[:, :R].copy()
    W_init = W71[:, :R].copy()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def residual(params):
        size_U = n * m * R
        size_V = m * p * R
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[i, j, k] = val - 1.0
        return res.flatten()

    start_time = time.time()
    try:
        # Use trust-reflect or similar if LM is too slow?
        # LM is good for small residuals.
        res = least_squares(residual, x0, method='lm', max_nfev=2000)
        if res.cost < 1e-12:
            size_U = n * m * R
            size_V = m * p * R
            U = res.x[:size_U].reshape(n*m, R)
            V = res.x[size_U:size_U+size_V].reshape(m*p, R)
            W = res.x[size_U+size_V:].reshape(p*n, R)
            return U, V, W
    except Exception:
        pass
    return None
