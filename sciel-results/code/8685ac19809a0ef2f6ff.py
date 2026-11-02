import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T."""
        # T_full[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T, atol=tol)
    
    def alternating_minimization(U, V, W, R, max_iter=200, tol=1e-8):
        """Run alternating least squares to minimize ||T - sum_r u_r⊗v_r⊗w_r||^2."""
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Update U: for each u, solve linear system
            # T[u,:,:] = sum_r U[u,r] * (V[v,r] * W[w,r])
            # So T[u,:,:] = U[u,:] @ (V * W) where (V*W) is reshaped
            # More precisely: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
            # For fixed V, W: T[u,:,:] = sum_r U[u,r] * (V[:,r] * W[:,r])
            # Let M_vw = V.T * W.T (shape R x mp x pn)
            # Then T[u,v,w] = sum_r U[u,r] * M_vw[r,v,w]
            # So T[u] = U[u,:] @ M_vw.reshape(R, -1)
            # Solve: U[u,:] @ M_vw.reshape(R, -1) = T[u].reshape(-1)
            
            M_vw = np.einsum('vr,wr->rvw', V, W)  # shape (R, mp, pn)
            M_vw_flat = M_vw.reshape(R, -1)  # shape (R, mp*pn)
            
            # For each u: U[u,:] @ M_vw_flat.T = T[u].reshape(-1)
            # This is a linear system: M_vw_flat.T @ U[u,:].T = T[u].reshape(-1)
            for u in range(n*m):
                rhs = T[u].reshape(-1)
                # Solve M_vw_flat.T @ x = rhs
                try:
                    U[u, :] = np.linalg.lstsq(M_vw_flat.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update V: for each v, solve linear system
            M_uw = np.einsum('ur,wr->ruw', U, W)  # shape (R, nm, pn)
            M_uw_flat = M_uw.reshape(R, -1)  # shape (R, nm*pn)
            
            for v in range(m*p):
                rhs = T[:, v, :].reshape(-1)
                try:
                    V[v, :] = np.linalg.lstsq(M_uw_flat.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update W: for each w, solve linear system
            M_uv = np.einsum('ur,vr->ruv', U, V)  # shape (R, nm, mp)
            M_uv_flat = M_uv.reshape(R, -1)  # shape (R, nm*mp)
            
            for w in range(p*n):
                rhs = T[:, :, w].reshape(-1)
                try:
                    W[w, :] = np.linalg.lstsq(M_uv_flat.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            if it % 10 == 0:
                T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(T_full - T)
                if err < tol:
                    break
        
        return U, V, W
    
    # Try ranks from 41 down to 35
    for R in range(41, 34, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 10:
                break
            
            np.random.seed(42 + R * 100 + restart)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            U, V, W = alternating_minimization(U, V, W, R, max_iter=150)
            
            if check_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return get_schoolbook()
