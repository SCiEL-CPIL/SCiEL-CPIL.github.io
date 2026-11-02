    def als_decompose(r, max_iters=150, seed=0):
        np.random.seed(seed)
        
        # SVD Initialization
        # Mode 1 unfolding
        T1 = T.transpose(0, 1, 2).reshape(n*m, -1)
        U_svd, _, _ = np.linalg.svd(T1, full_matrices=False)
        U = U_svd[:, :r].copy()
        
        # Mode 2 unfolding
        T2 = T.transpose(1, 0, 2).reshape(m*p, -1)
        V_svd, _, _ = np.linalg.svd(T2, full_matrices=False)
        V = V_svd[:, :r].copy()
        
        # Mode 3 unfolding
        T3 = T.transpose(2, 0, 1).reshape(p*n, -1)
        W_svd, _, _ = np.linalg.svd(T3, full_matrices=False)
        W = W_svd[:, :r].copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            M = np.zeros((m*p*p*n, r))
            for t in range(r):
                M[:, t] = np.kron(V[:, t], W[:, t])
            
            for a in range(n*m):
                rhs = T[a].flatten()
                U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Update V
            M = np.zeros((n*m*p*n, r))
            for t in range(r):
                M[:, t] = np.kron(U[:, t], W[:, t])
            
            for b in range(m*p):
                rhs = T[:, b, :].flatten()
                V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Update W
            M = np.zeros((n*m*m*p, r))
            for t in range(r):
                M[:, t] = np.kron(U[:, t], V[:, t])
            
            for c in range(p*n):
                rhs = T[:, :, c].flatten()
                W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            if it % 10 == 0 and check_decomposition(U, V, W, tol=1e-5):
                break
        
        # Try rounding to half-integers
        for scale in [1, 2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_decomposition(U_r, V_r, W_r, tol=1e-10):
                return U_r, V_r, W_r
        
        return None
