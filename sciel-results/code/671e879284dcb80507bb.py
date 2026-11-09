def try_random_rank43(n, m, p, deadline):
    R = 43
    np.random.seed(int(time.time() * 1000) % 10000)
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Scale to help convergence?
    # ALS
    for it in range(100):
        if time.time() > deadline - 1:
            break
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Check convergence
        if check(U, V, W, n, m, p):
            # Round
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            if check(U, V, W, n, m, p):
                return U, V, W
    return None
