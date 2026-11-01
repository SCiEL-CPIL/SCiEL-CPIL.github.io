def _search_rank13(time_budget=10.0):
    n, m, p = 2, 2, 4
    target_rank = 13
    T = np.ones((n, m, p))
    
    start_time = time.time()
    best_result = None
    
    for restart in range(200):
        if time.time() - start_time > time_budget - 5.0:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() - start_time > time_budget - 5.0:
                break
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    # Build 2 x rank matrix
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve M @ w = [1, 1]
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # For fixed (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                    # So: U[i*m+j, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) summed over k... 
                    # Actually: sum_k [sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = 1 for each k
                    # This is: for each k, U[i*m+j, :] · (V[j*p+k, :] * W[k*n+i, :]) = 1
                    # So we have p equations in rank unknowns
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # For fixed (j,k): sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
                    # So: for each i, V[j*p+k, :] · (U[i*m+j, :] * W[k*n+i, :]) = 1
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
        
        # Check convergence
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        error = np.max(np.abs(T - T_recon))
        
        if error < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if best_result is None or error < best_result[0]:
            best_result = (error, U.copy(), V.copy(), W.copy())
    
    if best_result is not None and best_result[0] < 1e-9:
        return {"U": best_result[1].tolist(), "V": best_result[2].tolist(), "W": best_result[3].tolist()}
    
    return None
