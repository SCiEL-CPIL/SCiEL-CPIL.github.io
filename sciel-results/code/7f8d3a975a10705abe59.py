    def try_rank(R, max_iters=20, num_restarts=50):
        """Try to find a rank-R decomposition using Alternating Least Squares"""
        if R >= best_rank:
            return None
        
        best_result = None
        best_obj = np.inf
        
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Initialize with random small integers
            rng = np.random.RandomState(R * 1000 + restart)
            U = rng.randint(-2, 3, size=(n * m, R)).astype(float)
            V = rng.randint(-2, 3, size=(m * p, R)).astype(float)
            W = rng.randint(-2, 3, size=(p * n, R)).astype(float)
            
            for _ in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Update U
                for a in range(n * m):
                    i = a // m
                    j = a % m
                    M = np.zeros((p, R))
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        M[k, :] = V[b, :] * W[c, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[a, :] = u_sol
                    except:
                        pass

                # Update V
                for b in range(m * p):
                    j = b // p
                    k = b % p
                    M = np.zeros((n, R))
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        M[i, :] = U[a, :] * W[c, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[b, :] = v_sol
                    except:
                        pass

                # Update W
                for c in range(p * n):
                    k = c // n
                    i = c % n
                    M = np.zeros((m, R))
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        M[j, :] = U[a, :] * V[b, :]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[c, :] = w_sol
                    except:
                        pass

                # Check convergence
                res = 0.0
                for a, b, c in triples:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    res += (val - 1.0) ** 2
                
                if res < 1e-10:
                    return U, V, W
                
                if res < best_obj:
                    best_obj = res
                    best_result = (U.copy(), V.copy(), W.copy())
            
            # End of restart iterations

        if best_result is not None:
            U_try, V_try, W_try = best_result
            if verify(U_try, V_try, W_try, tol=1e-6):
                return best_result
        
        return None
