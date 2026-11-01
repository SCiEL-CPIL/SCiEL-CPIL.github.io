def _try_rank10_als(deadline):
    rank = 10
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    best_result = None
    best_res = float('inf')
    
    max_trials = 50
    max_iters = 500
    
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break
            
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            U_new = np.zeros_like(U)
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        b_vec = np.zeros(p)
                        for k in range(p):
                            s_sum = 0.0
                            for s in range(rank):
                                if s != r:
                                    s_sum += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                            b_vec[k] = 1.0 - s_sum
                        y_vec = np.zeros(p)
                        for k in range(p):
                            y_vec[k] = V[j*p+k, r] * W[k*n+i, r]
                        num = np.dot(b_vec, y_vec)
                        den = np.dot(y_vec, y_vec)
                        if den > 1e-12:
                            U_new[i*m+j, r] = num / den
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for r in range(rank):
                for j in range(m):
                    for k in range(p):
                        b_vec = np.zeros(n)
                        for i in range(n):
                            s_sum = 0.0
                            for s in range(rank):
                                if s != r:
                                    s_sum += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                            b_vec[i] = 1.0 - s_sum
                        y_vec = np.zeros(n)
                        for i in range(n):
                            y_vec[i] = U[i*m+j, r] * W[k*n+i, r]
                        num = np.dot(b_vec, y_vec)
                        den = np.dot(y_vec, y_vec)
                        if den > 1e-12:
                            V_new[j*p+k, r] = num / den
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for r in range(rank):
                for k in range(p):
                    for i in range(n):
                        b_vec = np.zeros(m)
                        for j in range(m):
                            s_sum = 0.0
                            for s in range(rank):
                                if s != r:
                                    s_sum += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                            b_vec[j] = 1.0 - s_sum
                        y_vec = np.zeros(m)
                        for j in range(m):
                            y_vec[j] = U[i*m+j, r] * V[j*p+k, r]
                        num = np.dot(b_vec, y_vec)
                        den = np.dot(y_vec, y_vec)
                        if den > 1e-12:
                            W_new[k*n+i, r] = num / den
            W = W_new
            
            # Check residual
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_res = max(max_res, abs(s - 1.0))
            
            if max_res < best_res:
                best_res = max_res
                if max_res < 1e-9:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
            if max_res < 1e-12:
                break
                
    if best_res < 1e-9:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None
