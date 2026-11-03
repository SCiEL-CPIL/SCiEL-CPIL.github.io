    def try_rank(R, seed=42):
        """Try to find a rank-R decomposition using ALS"""
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return False
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.01
        V = np.random.randn(m*p, R) * 0.01
        W = np.random.randn(p*n, R) * 0.01
        
        # Precompute flattened views of T for faster access
        # T shape: (n*m, m*p, p*n)
        T_flat_u = T.reshape(n*m, m*p*p*n)  # For updating U
        T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n) # For updating V
        T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p) # For updating W

        prev_residual = np.inf
        for it in range(20): # Reduced iterations to save time, increased if needed
            if time.time() > deadline:
                return False
            
            # Update U
            # Compute VW_flat: (R, m*p*p*n)
            VW = np.einsum('br,cr->bc', V, W) # This is wrong, V is (m*p, R), W is (p*n, R)
            # V[:, r] is (m*p,). W[:, r] is (p*n,).
            # Outer product is (m*p, p*n).
            # We need to stack these for all r.
            # V shape (m*p, R). W shape (p*n, R).
            # V[:,:,None] * W[None,:,:] -> (m*p, p*n, R).
            # We want (R, m*p, p*n) or (R, m*p*p*n).
            VW_3d = V[:, :, np.newaxis] * W[np.newaxis, :, :] # (m*p, p*n, R)
            VW_flat = VW_3d.transpose(2, 0, 1).reshape(R, m*p*p*n) # (R, m*p*p*n)
            
            # M_u = VW_flat @ VW_flat.T (R x R)
            # N_u = T_flat_u @ VW_flat.T (n*m x R)
            M_u = VW_flat @ VW_flat.T
            N_u = T_flat_u @ VW_flat.T
            # Add regularization to handle singularities
            M_u += 1e-8 * np.eye(R)
            try:
                U = N_u @ np.linalg.inv(M_u)
            except np.linalg.LinAlgError:
                U = N_u @ np.linalg.pinv(M_u)
                
            # Update V
            UW_3d = U[:, :, np.newaxis] * W[np.newaxis, :, :] # (n*m, p*n, R)
            UW_flat = UW_3d.transpose(2, 0, 1).reshape(R, n*m*p*n) # (R, n*m*p*n)
            
            M_v = UW_flat @ UW_flat.T
            N_v = T_flat_v @ UW_flat.T
            M_v += 1e-8 * np.eye(R)
            try:
                V = N_v @ np.linalg.inv(M_v)
            except np.linalg.LinAlgError:
                V = N_v @ np.linalg.pinv(M_v)

            # Update W
            UV_3d = U[:, :, np.newaxis] * V[np.newaxis, :, :] # (n*m, m*p, R)
            UV_flat = UV_3d.transpose(2, 0, 1).reshape(R, n*m*m*p) # (R, n*m*m*p)
            
            M_w = UV_flat @ UV_flat.T
            N_w = T_flat_w @ UV_flat.T
            M_w += 1e-8 * np.eye(R)
            try:
                W = N_w @ np.linalg.inv(M_w)
            except np.linalg.LinAlgError:
                W = N_w @ np.linalg.pinv(M_w)
            
            # Compute residual
            # Reconstruct T_approx
            # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Reshape U, V, W to (dim, R)
            # Use einsum
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_approx)
            
            if abs(prev_residual - residual) < 1e-12:
                break
            prev_residual = residual
            
            if residual < 1e-6:
                if R < best_rank:
                    best_rank = R
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                return True
        
        # Check final residual
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T - T_approx)
        if residual < 1e-6:
            if R < best_rank:
                best_rank = R
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            return True
        
        return False
