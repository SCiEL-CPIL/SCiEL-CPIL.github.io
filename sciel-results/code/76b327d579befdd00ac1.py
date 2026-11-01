def _try_rank10():
    """Try to find a rank-10 decomposition."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = 10
    num_vars = n*m*rank + m*p*rank + p*n*rank  # 4*10+6*10+6*10 = 160
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    
    def compute_residual(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        return T_decomp.flatten() - T_flat
    
    # Try multiple random initializations
    best_result = None
    for trial in range(20):
        np.random.seed(trial)
        params = np.random.randn(num_vars)
        
        # Use ALS for a few iterations
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        for it in range(50):
            # Update U
            for u in range(n*m):
                # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
                # For fixed u: linear in U[u,:]
                # Build system: for each (v,w), sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
                A_mat = np.zeros((m*p*p*n, rank))
                b_vec = T[u, :, :].flatten()
                idx = 0
                for v in range(m*p):
                    for w in range(p*n):
                        A_mat[idx, :] = V[:, :] * W[w, :]  # element-wise, shape (rank,)
                        idx += 1
                U[u, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V
            for v in range(m*p):
                A_mat = np.zeros((n*m*p*n, rank))
                b_vec = T[:, v, :].flatten()
                idx = 0
                for u in range(n*m):
                    for w in range(p*n):
                        A_mat[idx, :] = U[u, :] * W[w, :]
                        idx += 1
                V[v, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W
            for w in range(p*n):
                A_mat = np.zeros((n*m*m*p, rank))
                b_vec = T[:, :, w].flatten()
                idx = 0
                for u in range(n*m):
                    for v in range(m*p):
                        A_mat[idx, :] = U[u, :] * V[v, :]
                        idx += 1
                W[w, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check
            T_decomp = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.linalg.norm(T_decomp - T)
            if err < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
