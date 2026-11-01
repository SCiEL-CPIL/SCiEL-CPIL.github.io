def _find_2x2x3_rank10(deadline):
    n, m, p = 2, 2, 3
    rank = 10
    best_error = float('inf')
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        
        # Initialize with small integers
        U = np.random.randint(-2, 3, (4, rank)).astype(float)
        V = np.random.randint(-2, 3, (6, rank)).astype(float)
        W = np.random.randint(-2, 3, (6, rank)).astype(float)
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline - 20:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B_mat = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B_mat[r, k] = V[j*p + k, r] * W[k*n + i, r]
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C_mat = np.zeros((rank, n*m))
                    for r in range(rank):
                        for i in range(n):
                            C_mat[r, i*m + j] = U[i*m + j, r] * W[k*n + i, r]
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(C_mat, np.ones(n*m), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    D_mat = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D_mat[r, j] = U[i*m + j, r] * V[j*p + k, r]
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(D_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-14:
                break
    
    if best_solution is not None and best_error < 1e-10:
        U, V, W = best_solution
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify_2x2x3(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Round to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if _verify_2x2x3(U_r, V_r, W_r):
            return U_r, V_r, W_r
    
    return None, None, None
