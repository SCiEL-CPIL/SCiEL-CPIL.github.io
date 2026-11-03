def _find_223_rank10(max_time=30):
    import time
    start = time.time()
    n, m, p = 2, 2, 3
    r = 10
    
    def _residual(params):
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        res = []
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c]
                        expected = 1.0 if j == jp else 0.0
                        res.append(s - expected)
        return np.array(res)
    
    from scipy.optimize import least_squares
    
    best_cost = np.inf
    best_params = None
    
    for trial in range(50):
        if time.time() - start > max_time:
            break
        
        params0 = np.random.randn(160) * 0.1
        try:
            result = least_squares(_residual, params0, max_nfev=1000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x
        except:
            pass
    
    if best_params is not None and best_cost < 1e-10:
        U = best_params[0:40].reshape(4, 10)
        V = best_params[40:100].reshape(6, 10)
        W = best_params[100:160].reshape(6, 10)
        return U, V, W
    
    return None
