def _try_random_exact(rank, n, m, p, n_seeds=100):
    if time.time() - _START_TIME > _TIMEOUT:
        return None
    
    for seed in range(n_seeds):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(200):
            if time.time() - _START_TIME > _TIMEOUT:
                return None
            
            U_new = np.zeros((n*m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(rank)] for k in range(p)])
                    b = np.ones(p)
                    sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j] = sol
            
            V_new = np.zeros((m*p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(rank)] for i in range(n)])
                    b = np.ones(n)
                    sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k] = sol
            
            W_new = np.zeros((p*n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(rank)] for j in range(m)])
                    b = np.ones(m)
                    sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i] = sol
            
            U, V, W = U_new, V_new, W_new
            
            if _verify_float(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-10):
                if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U.tolist(), V.tolist(), W.tolist()
    
    return None
